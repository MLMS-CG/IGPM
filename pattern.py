import numpy as np
import trimesh
from tqdm import tqdm
from panel import Panel
from symmetry import compute_symmetric_control_points_positions_p
import torch
import config

if config.setting["gpu_enable"]=="on":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]

#which element in subsets belong to pool, get its index
def findSubset(pool,subsets):
    for i in range(len(subsets)):
        elements= set(subsets[i])
        if elements.issubset(set(pool)):
            return i
    return -1

def get_global_indices(part,template):
    """
    Afte using trimesh split, each part/panel is an independent trimesh object, so the vertex indices are now locally 
    assigned for each part like [0,1,2,3...], get_global_indices aims at retriving their global vertex indices in 
    the whole pattern. 
    
    #attention: this is coordinates based, so the panels should not overlap, otherwise can be ambigous.
    """
    global_indices=[]
    mat=template.vertices
    
    for i in tqdm(range(len(part.vertices))): 
        row=part.vertices[i]
        idj=findByRow(mat, row) # return inside a list [indice]
        global_indices.append(idj[0])
    return global_indices

def reorder_feature_points(feature_indices, global_indices):
    """
    reorder the feature vertices(control points in the paper) grouped by panels and make it same to the order of panels 
    in the pattern after split. 
    """
    part_feature_order=[] # reorder the feature points to make them consistent to the panels order
    for i,global_indices_part in enumerate(global_indices):
        for j, feature_indices_part in enumerate(feature_indices):
            if np.isin(feature_indices_part, global_indices_part, invert=False).any():
                #print(i,j)
                part_feature_order.append(j)
                break
        
    feature_points_reorderd = [feature_indices[i] for i in part_feature_order]
    
    return feature_points_reorderd
    


def get_parttern_panels(parts, pattern, control_points_grouped_by_panel, feature_reversed, modes):
    """
    get a list of Panel instances together with a list of panel vertices' global indices, both in the split order of Trimesh.
    new_orders: the order mapping control_points_grouped_by_panel (k-th) to parts (i-th) 
    """
    panels=[]
    global_part_indices=[]
    new_orders=[]
    
    for i,part in enumerate(parts):
        global_part_indice = get_global_indices(part, pattern) #global indices by part/panel
        global_part_indices.append(global_part_indice)
        k=findSubset(global_part_indice,control_points_grouped_by_panel)
        
        #print("k is",k)
        
        new_orders.append(k)  #i-th part corresponds to k-th cts.
        
        global_con_indice=control_points_grouped_by_panel[k] # global constraint indices of control points
        con_reverse=feature_reversed[k]

        mode=modes[k]
        panel=Panel(part, global_part_indice, global_con_indice, con_reverse, mode)
        panels.append(panel)
    
    return global_part_indices, panels, new_orders    
    

def from_symmetry(inter_symmetry_info, intra_symmetry_info):
    """
    Based on the results of inter and intra symmetry detection, deduce the effective control points that
    would be used during the optimzation. 
    """
    feature_points_id, symmetric_feature_points_id, _, feature_reversed = inter_symmetry_info    
    self_symmetric_feature_points_id, _, _, self_symmetric_pattern_flag = intra_symmetry_info
    
    
    control_points_grouped_by_panel=[]  # added according the inter symmetry pair order 
    effective_points_indices=[]  # effective control points in paper

    modes=[]  # to choose the mode of deformation #TODO remove modes for good
    for i in range(len(feature_points_id)):  
        # for control_points_grouped_by_panel: add the in the order of feature_points and 
        #then the paried symmetric_feature_point if the symmetry exists
        control_points_grouped_by_panel.append(feature_points_id[i])    #feature_points_id grouped by panel
        modes.append(True)
        if i < (len(symmetric_feature_points_id)): # valid i, meaing there is symmetric counterpart 
            control_points_grouped_by_panel.append(symmetric_feature_points_id[i])
            modes.append(True)

        # fill the list of effective control points
        if self_symmetric_pattern_flag[i]: # leverage intra-symmetry
            for idx in feature_points_id[i]:
                #if the point is not in the symmetry list, then just directly extract its coordinates from feature points
                if idx not in self_symmetric_feature_points_id[i]: 
                        effective_points_indices.append(idx)
        else:
            effective_points_indices+=feature_points_id[i]
        
    return control_points_grouped_by_panel, effective_points_indices, feature_reversed, modes 
   

def compute_pattern_constraints(template, patterns_loaded, control_points_grouped_by_panel, feature_points_indices, feature_reversed, modes):     
    parts=template.split(only_watertight=False) # flat 2d pattern mesh is composed of panels, so we need to split it. 
          
    pattern_infos = get_parttern_panels(parts,template,control_points_grouped_by_panel, feature_reversed, modes)
    
    u=np.array(patterns_loaded.vertices)[:,:-1] # here we follow the notation of Arcsim, u indicates the position of vertices in 2d material space
    
    # set inital feature tensor
    u_fp=u[feature_points_indices,:]
    #u_tensor = torch.from_numpy(u_fp).float().to(device)  # force to be float
    u_tensor = torch.from_numpy(u_fp).to(device)

    u_tensor.requires_grad_(True) #autograd should record operations on u_hem for backpropagation


    return pattern_infos, u_tensor

def warp_all_panels(num_verts, global_part_indices, panels, control_points):
    """
    apply deformation to every parts/panels and assemble their results for a new pattern mesh
    """
    warped_vertices=torch.zeros((num_verts,2),device=device).double()   #todo
    #warped_vertices=torch.zeros((num_verts,2),device=device)
    
    # panels and control_points follow the order of split 
    for i, panel in enumerate(panels): 
        new_ctrls=control_points[i]  # it's already 2D now
        x=panel.mean_value_coordinates(new_ctrls)
        
        warped_vertices[panel.global_part_indices]=x
    return warped_vertices


def get_new_pattern(pattern_mesh, u_ten, feature_points_indices, panels_info, inter_symmetry, intra_symmetry):
    """
    update the pattern, with the new coordinates in u_ten (effective ponits). The pattern mesh is 
    restored while respecting the symmetry info obtained from the function from_symmetry
    """
    feature_points_id, symmetric_feature_points_id, symmetric_feature_points_tm, _=inter_symmetry    
    self_symmetric_feature_points_id, self_feature_points_id, self_symmetric_feature_points_tm, self_symmetric_pattern_flag=intra_symmetry
    
    
    num_verts=pattern_mesh.vertices.shape[0]
    
    # effective points--->contril points
    feature_points_positions=u_ten # 2D for symmetry recovery
    control_points_positions=[] 

    
    #[[2, 410, 421, 425, 21, 26], [130, 125, 91, 80, 55], [237, 219, 325, 337, 341, 232]]
    for i in range(len(feature_points_id)):  
       
        # True if there is self symmetry, get all the control points coords in this pattern by intra-sym firstly
        if self_symmetric_pattern_flag[i]: 
            
            
            self_ctrls_indices=self_feature_points_id[i]
            self_symmetric_ctrls_indices=self_symmetric_feature_points_id[i]
            
            control_points_positions_p=[]
            for idx in feature_points_id[i]:
                #if the point is not in the self symmetry list, then just directly extract its coordinates from feature points
                if idx not in self_symmetric_feature_points_id[i]: 
                    
                    idx_in_feature_tensor=feature_points_indices.index(idx)
                    control_points_positions_p.append(feature_points_positions[idx_in_feature_tensor:idx_in_feature_tensor+1]) # direct feature points, and keep dim by val_keep = val[:,1:2,:,:]
                
                # pts that can be obtained by intra symmetry
                else:
                    
                     # get the vertex indice, same in the self feature points and the self symmetry feature pts                      
                    idx_=self_ctrls_indices[self_symmetric_ctrls_indices.index(idx)] #vertex index of the symmetry source
                    
                    
                    #extract the source self feature point from feature point tensor
                    idx_in_feature_tensor=feature_points_indices.index(idx_)
                    self_feature_points_position= feature_points_positions[idx_in_feature_tensor:idx_in_feature_tensor+1]
                    
                    #use intra_symmetry to compute the target corner point
                    control_point_by_intra_symmetry=compute_symmetric_control_points_positions_p(self_feature_points_position,self_symmetric_feature_points_tm[i])
                    
                    control_points_positions_p.append(control_point_by_intra_symmetry) #1*3
                                                    
            control_points_positions_p=torch.cat(control_points_positions_p,0)  #n*3  
        
        #if no self symmetry just extract the vetices positions from feature tensor                                            
        else: 
            indexing= [feature_points_indices.index(idx)  for idx in feature_points_id[i]]
            control_points_positions_p=feature_points_positions[indexing]  #n*3
        
        
        control_points_positions.append(control_points_positions_p) 

        # if the feature_points do not have the inter counterparts, then just carry on to next one
        #otherwise, also get the inter symmetric_control_points_positions
        if i > len(symmetric_feature_points_id)-1: # not valid i 
            continue   
        
        # deal with the inter symmetry if any
        symmetric_control_points_positions_p=compute_symmetric_control_points_positions_p(control_points_positions_p,symmetric_feature_points_tm[i]) #The inter symmetric pattern computed here
        control_points_positions.append(symmetric_control_points_positions_p)
    
    #warp is done in 2D, so the control_pts should be cut 
    global_part_indices, panels, new_orders = panels_info
    
    #restored control_points_positions follow the symmetry order, so reordered to be split panels order
    control_points_coords = [control_points_positions[i] for i in new_orders]    
    warped_vertices=warp_all_panels(num_verts, global_part_indices, panels, control_points_coords)
    
    return warped_vertices
        
if __name__ == "__main__":

    #-----------unit test-------------
    from boundary_utils import get_boundary_edges,get_grouped_boundaries_anticlc, get_corner_points,control_points,assembling_info
    from symmetry import find_inter_symmetry, find_intra_symmetry

    cloth_2d="./meshes/tshirtm.obj"
    cloth_3d="./meshes/tshirtT.obj"
    cloth_2d_3d="./meshes/tshirt.obj"
    load_pattern_path=cloth_2d


    patterns_loaded=trimesh.load(load_pattern_path,process=False)
    

    template_patterns_path=cloth_2d
    template=trimesh.load(template_patterns_path,process=False)
    parts=template.split(only_watertight=False)
    edges=get_boundary_edges(template)
    
    pattern_boundaries=get_grouped_boundaries_anticlc(template,get_corner_points) #only the corner points
    print(pattern_boundaries)

    nodes=trimesh.load(cloth_3d,process=False).vertices
    nb_nodes=len(nodes)
    print(nb_nodes)

    
    n2v_dic, v2n_dic, join_points=assembling_info(cloth_2d_3d,nb_nodes)
    pattern_boundaries=control_points(edges,pattern_boundaries,join_points)
    print(pattern_boundaries)
    
    inter_symmetry=find_inter_symmetry(template, pattern_boundaries)
    feature_points_id=inter_symmetry[0]
    print(inter_symmetry[0])
    print(inter_symmetry[1])

    print("*******************")
    intra_symmetry=find_intra_symmetry(template,feature_points_id)
    print(intra_symmetry[0])
    print(intra_symmetry[1])
    

    #Test: restore pattern mesh and preserve symmetry 
    control_points_grouped_by_panel, feature_points_indices, feature_reversed, modes=from_symmetry(inter_symmetry,intra_symmetry)
    pattern_infos, u_tensor=compute_pattern_constraints(template, patterns_loaded, control_points_grouped_by_panel, feature_points_indices, feature_reversed, modes)     
     
     

    print("u_ten: ",u_tensor.dtype)
    
    warped_vertices=get_new_pattern(template, u_tensor, feature_points_indices, pattern_infos, inter_symmetry, intra_symmetry)
     
    print(warped_vertices.shape)
    print(inter_symmetry[1][0])
    print(warped_vertices[inter_symmetry[1][0]])
     
    print("done")
    


    """        
     V=warped_vertices.clone().detach().numpy()
     F=template.faces
     new_patterns=trimesh.Trimesh(V,F,process=False)
     new_patterns_data=trimesh.exchange.obj.export_obj(new_patterns)
     with open("test_mesh.obj","w") as f:
             f.write(new_patterns_data)  
             
    """
