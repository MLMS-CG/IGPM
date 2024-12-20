import os
import fnmatch
import trimesh
import numpy as np
import json
from boundary_utils import get_boundary_edges,get_grouped_boundaries_anticlc, get_corner_points,control_points,assembling_info
from pattern import reorder_feature_points, get_global_indices

def removecfs(remove_list, pattern_boundaries):
    new_pattern_boundaries=[]
    for panel_bounds in  pattern_boundaries:
        panel_bounds_filtered = [x for x in panel_bounds if x not in remove_list]
        new_pattern_boundaries.append(panel_bounds_filtered)
    return new_pattern_boundaries


def permut(l,t):
    return [l[i] for i in t]

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)
        
def count_files(directory, prefix):
    lst=os.listdir(directory)
    cnt_list = len(fnmatch.filter(lst, prefix+'*')) 
    return cnt_list

        
def save_pattern(output_path, faces, vertices, epoch, device, save_flag="v"):

        name={'v':'garmentm','n':'garment'}
        #save 2d patterns
        F=faces
        if device=="cpu":
            verts=vertices.clone().detach().numpy()
            if save_flag=="v":
                V=np.concatenate((verts,np.zeros((vertices.shape[0],1))),-1) #save pattern obj requiring add one dim 
            else:
                V=verts
        else:
            verts=vertices.cpu().clone().detach().numpy()
            if save_flag=="v":
                V=np.concatenate((verts,np.zeros((vertices.shape[0],1))),-1)
            else:
                V=verts
                
        new_patterns=trimesh.Trimesh(V,F,process=False)
        new_patterns_data=trimesh.exchange.obj.export_obj(new_patterns)
        #print("save m for epoch {}".format(epoch))
        with open(output_path+"/"+name[save_flag]+"_epoch_{}.obj".format(epoch),"w") as f:
                f.write(new_patterns_data)

# To get the control pts indices and eventually add/remove certain from them.
# The returned are reordered to be consistent with the order of panels after pattern mesh being split.
def get_ordered_bounds(cloth_2d, cloth_3d, cloth_2d3d, add_list=[], remove_list=[]):
    #pattern
    pattern_mesh=trimesh.load(cloth_2d,process=False)

    feature_vertices=get_grouped_boundaries_anticlc(pattern_mesh, get_corner_points)
    parts=pattern_mesh.split(only_watertight=False,adjacency=None)

    global_indices=[]
    for part in parts:
        global_indices.append(get_global_indices(part,pattern_mesh))
    
    garment=trimesh.load(cloth_3d,process=False)
    nodes=garment.vertices
    
    nb_nodes=len(nodes)
    n2v_dic, v2n_dic, join_points = assembling_info(cloth_2d3d, nb_nodes)
    #v2n_arr=np.array(list(v2n_dic.values()))

    pattern_edges=get_boundary_edges(pattern_mesh)
    
    merged_list = list(set(join_points) | set(add_list))
    
    feature_vertices=control_points(pattern_edges,feature_vertices,merged_list) 
        
    feature_vertices=removecfs(remove_list, feature_vertices) # optional
    
    # align the order of grouped control points(per panel) same to the panel order of trimesh split function
    feature_points_reordered=reorder_feature_points(feature_vertices, global_indices)
    
    return feature_points_reordered



# Get the control points.
def get_pattern_boundaries(cloth_2d,cloth_3d,cloth_2d_3d): 
    pattern_mesh=trimesh.load(cloth_2d,process=False)
    edges=get_boundary_edges(pattern_mesh)

    pattern_boundaries=get_grouped_boundaries_anticlc(pattern_mesh, get_corner_points) #only the corner points

    verts=trimesh.load(cloth_3d,process=False).vertices
    nb_nodes=len(verts)

    n2v_dic, v2n_dic, join_points=assembling_info(cloth_2d_3d,nb_nodes)

    feature_vertices=control_points(edges,pattern_boundaries,join_points)
    return feature_vertices


