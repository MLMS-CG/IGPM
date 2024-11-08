from scipy.linalg import orthogonal_procrustes
import numpy as np
import trimesh
import math
import torch
import config

def compute_procrustes_transformation_matrix(vertices, feature_points_idx_1, feature_points_idx_2):
    
    vertices=vertices[:,:2] #modified to be 2D 
      
    # Compute the coordinates of the first list of boundary vertices
    bound_coord1 = np.array(vertices[feature_points_idx_1])
    # Remove the translation part
    pre_trans = bound_coord1.sum(axis=0)/len(bound_coord1)
    tmp = np.repeat([pre_trans],repeats=len(bound_coord1), axis=0)
    bound_coord1 = bound_coord1 - tmp

    # Compute the coordinates of the second list of boundary vertices
    bound_coord2 = np.array(vertices[feature_points_idx_2])
    # Remove the translation part
    post_trans = bound_coord2.sum(axis=0)/len(bound_coord2)
    tmp = np.repeat([post_trans],repeats=len(bound_coord2), axis=0)
    bound_coord2 = bound_coord2 - tmp

    # Compute the transformation matrix
    R, sca = orthogonal_procrustes(bound_coord2, bound_coord1) # coord2*R-->coord1
    # Compute the matching score
    matching_score = np.linalg.norm(np.matmul(R,bound_coord1.T)-bound_coord2.T)
        
    # Compute the transformation matrix that includes the rotation and the two translations
    temp_r = np.eye(3)
    temp_r[0][0:2] = R[0][0:2]
    temp_r[1][0:2] = R[1][0:2]
    
    temp_pre_trans = np.eye(2)
    temp_pre_trans = np.append(temp_pre_trans,np.array([-pre_trans]).T,axis=1)
    temp_pre_trans = np.append(temp_pre_trans,np.array([[0, 0, 1]]),axis=0)
    
    temp_post_trans = np.eye(2)
    temp_post_trans = np.append(temp_post_trans,np.array([post_trans]).T,axis=1)
    temp_post_trans = np.append(temp_post_trans,np.array([[0, 0, 1]]),axis=0)
    
    
    tm = np.matmul(temp_r,temp_pre_trans)
    tm = np.matmul(temp_post_trans,tm)

    # Return the matching and the transformation matrix
    return matching_score, tm

def angle(x,y):
    """
    angle of two vectors
    """
    l_x=np.sqrt(x.dot(x))
    l_y=np.sqrt(y.dot(y))
    # get cos
    cos_=x.dot(y)/(l_x*l_y)
    # get the angle
    angle_hu=np.arccos(cos_)
    #print(angle_hu/np.pi*180)

    if angle_hu>np.pi/2:
        #angle_hu-=np.pi/2
        angle_hu=np.pi-angle_hu
    return angle_hu


def find_intra(bound_coord1,bound_coord2):
    """
    intra symmetry(mirror)
    """
    #https://blog.csdn.net/qq_38418182/article/details/126875088
    #https://www.cnblogs.com/softhal/p/5648463.html
    
    pre_trans = bound_coord1.sum(axis=0)/len(bound_coord1)
    post_trans = bound_coord2.sum(axis=0)/len(bound_coord2)

    center=(pre_trans+post_trans)/2
    vec=pre_trans-post_trans
    #print("vec is :", vec)
    theta=angle(np.array([1,0]),vec)
    if vec[0]*vec[1]>0:
        theta*=-1

    print("theta is: ",theta)

    # Compute the global coords to local coords matrix
    temp_pre_trans = np.eye(2)
    temp_pre_trans = np.append(temp_pre_trans,np.array([-center]).T,axis=1)
    temp_pre_trans = np.append(temp_pre_trans,np.array([[0, 0, 1]]),axis=0)

    temp_r = np.eye(3)
    temp_r[0][0] = np.cos(theta)
    temp_r[0][1] = -np.sin(theta)
    temp_r[1][0] = np.sin(theta)
    temp_r[1][1] = np.cos(theta)
    tm_g2l = np.matmul(temp_r,temp_pre_trans)

    # reflection
    R=np.array([[-1.0 , 0.0],
     [ 0.0 ,1.0]])
    flip_r = np.eye(3)
    flip_r[0][0:2] = R[0][0:2]
    flip_r[1][0:2] = R[1][0:2]

    # Compute the local back to gobal matrix
    temp_post_trans = np.eye(2)
    temp_post_trans = np.append(temp_post_trans,np.array([center]).T,axis=1)
    temp_post_trans = np.append(temp_post_trans,np.array([[0, 0, 1]]),axis=0)

    temp_r = np.eye(3)
    temp_r[0][0] = np.cos(theta)
    temp_r[0][1] = np.sin(theta)
    temp_r[1][0] = -np.sin(theta)
    temp_r[1][1] = np.cos(theta)
    tm_l2g = np.matmul(temp_post_trans,temp_r)

    #final tm
    tm = tm_l2g@flip_r@tm_g2l
    return tm




def compute_procrustes_transformation_matrix_intra(vertices, feature_points_idx_1, feature_points_idx_2):	
    	
    vertices=vertices[:,:2] #modified to be 2D 	
      	
    # Compute the coordinates of the first list of boundary vertices	
    bound_coord1 = np.array(vertices[feature_points_idx_1])	
    bound_coord2 = np.array(vertices[feature_points_idx_2])	
    tm= find_intra(bound_coord1,bound_coord2)	# get the transformarion matrix
    	
    recon_bound_coord2=(tm @ np.concatenate((bound_coord1.T,np.ones((1,len(feature_points_idx_1)))),axis=0))[:-1]	
    matching_score = np.linalg.norm(recon_bound_coord2-bound_coord2.T)	    	
    # Return the matching and the transformation matrix	
    return matching_score, tm

def find_symmetry_for_one_boundary(vertices, feature_points_idx_1, feature_points_idx_2):
    best_matching_score = None
    for i in range(0, len(feature_points_idx_2)):  #feature_points_idx_2 keep rolling
        matching_score, TM = compute_procrustes_transformation_matrix(vertices, feature_points_idx_1, feature_points_idx_2)
        # Check if the matching score is better
        if best_matching_score == None or best_matching_score > matching_score:
            best_matching_score = matching_score
            best_TM = TM
            # The = operator is cloning the list.
            # Any change on feature_points_idx_2 will modify best_feature_points_idx_2
            # So we need to use the syntaxe ...=list(...)
            best_feature_points_idx_2 = list(feature_points_idx_2)
        # Do a rolling of the second list of boundary vertices
        feature_points_idx_2.append(feature_points_idx_2.pop(0))

    return best_matching_score, best_TM, best_feature_points_idx_2


def find_self_symmetry_for_one_boundary(vertices, feature_points_idx):
    best_matching_score = None
    for i in range(0, len(feature_points_idx)):  #feature_points_idx keep rolling
        # For finding self-symmetry, we cut the boundary into two
        feature_points_idx_1 = feature_points_idx[0:math.floor(len(feature_points_idx)/2)]
        feature_points_idx_2 = feature_points_idx[math.ceil(len(feature_points_idx)/2):len(feature_points_idx)]
        
        feature_points_idx_2 = list(reversed(feature_points_idx_2))
        # Find the symmetry between these two pieces
        #matching_score, TM = compute_procrustes_transformation_matrix(vertices, feature_points_idx_1, feature_points_idx_2) #TODO
        matching_score, TM = compute_procrustes_transformation_matrix_intra(vertices, feature_points_idx_1, feature_points_idx_2)
        
        # Check if the matching score is better
        if best_matching_score == None or best_matching_score > matching_score:
            best_matching_score = matching_score
            best_TM = TM
            # The = operator is cloning the list.
            # Any change on feature_points_idx_2 will modify best_feature_points_idx_2
            # So we need to use the syntaxe ...=list(...)
            best_feature_points_idx_1 = list(feature_points_idx_1)
            best_feature_points_idx_2 = list(feature_points_idx_2)
        # Do a rolling of the list of boundary vertices
        feature_points_idx.append(feature_points_idx.pop(0))

    return best_matching_score, best_TM, best_feature_points_idx_1, best_feature_points_idx_2


def find_inter_symmetry(mesh, pattern_boundaries):
    """
    Compute the symmetry among the control points on pattern boundaries, which are grouped per panel
    All vertex indices are defined in the full pattern mesh
    """

    # List of the indices of the feature points
    feature_points_id = []

    # Indices of the feature points' counterparts which are symmetric to them:
    #  - symmetric_feature_points_id[0] is symmetric with feature_points_id[0]
    #  - symmetric_feature_points_id[1] is symmetric with feature_points_id[1]
    #  - etc.
    symmetric_feature_points_id = []

    # The transformation matrix to obtain the position of the symmetric feature point
    # , which includes the rotation and pre and post translation
    symmetric_feature_points_tm = []
    
    feature_reversed=[] # register 
    
     
    symmetric_pattern = [False] * len(pattern_boundaries)
    
    for i in range(len(pattern_boundaries)):
        if symmetric_pattern[i] == True:
            continue
            
        # Find inter-symmetry
        for j in range(i+1,len(pattern_boundaries)):
            # If the number of points is different, it cannot be symmetric
            if len(pattern_boundaries[i]) != len(pattern_boundaries[j]):
                continue
            # If the pattern has been already found as symmetric, we go to the next one
            if symmetric_pattern[j] == True:
                continue

            # Compute matching with reverse order of the feature points (anticlc and clc).
            reverse_flag=True
            matching_score, TM, boundary_j = find_symmetry_for_one_boundary(
                                                                            mesh.vertices,
                                                                            pattern_boundaries[i],
                                                                            list(reversed(pattern_boundaries[j])))
                        
            # If the matching score is below a certain threshold
            if matching_score < 15e-3: #10e-3:#5*1e-3:

                print("found matching between", str(i), " and ", str(j))
                # Set a flag to indicate that the boundary j has been found symmetric to boundary i
                symmetric_pattern[i] = True
                symmetric_pattern[j] = True
                
                
                symmetric_feature_points_id.append(boundary_j)
                # Save the index of feature point that is used for the computing the position of the symmetric feature point
                feature_points_id.append(pattern_boundaries[i])   
                # Save the transformation matrix for computing the position of the symmetric feature point
                symmetric_feature_points_tm.append(TM)
                
                feature_reversed.append(False)
                feature_reversed.append(reverse_flag)
                
                break #Once one to one symmetry is found
    
    #  if there remains some pts no symmetry is found, add them to feature points                 
    for idx in range(len(pattern_boundaries)):
        if symmetric_pattern[idx] == False:
                v=pattern_boundaries[idx]
                if v not in feature_points_id:
                    feature_points_id.append(v)
                    feature_reversed.append(False)
                    
    return feature_points_id, symmetric_feature_points_id, symmetric_feature_points_tm, feature_reversed

def find_intra_symmetry(mesh, feature_points_id, switch=True):
    """
    find intra-symmetry for "feature_points_id" to reduce further the feature points number 
    #switch: Bool True consider the intra sym -- False do not 
    """
    self_symmetric_feature_points_id=[]
    self_feature_points_id=[]
    self_symmetric_feature_points_tm=[]
    self_symmetric_pattern=[False]*len(feature_points_id)  # indicates if this group of feature points has intra symmetry
    
    if not switch:
        return self_symmetric_feature_points_id, self_feature_points_id, self_symmetric_feature_points_tm, self_symmetric_pattern

    for i in range(len(feature_points_id)):
        # Find intra-symmetry
        self_match_score, self_TM, self_feature_points_idx_1, self_feature_points_idx_2 = find_self_symmetry_for_one_boundary(mesh.vertices,
                                                                                                feature_points_id[i])
        print("self_mact_score of pattenr ",i ,"is ",self_match_score)

        #if self_match_score < 0.01:#0.005:
        if self_match_score < 0.05:#0.01 #0.005 # depending on the detection sensitivity  

            #print("found self matching for", str(i))
            self_symmetric_pattern[i] = True
            
            # A new symmetric feature point has been found
            self_symmetric_feature_points_id.append(self_feature_points_idx_2)
            # Save the index of feature point that is used for the computing the position of the symmetric feature point
            self_feature_points_id.append(self_feature_points_idx_1)
            # Save the transformation matrix for computing the position of the intra symmetric feature point
            self_symmetric_feature_points_tm.append(self_TM)
            
        else: #if no intra symmetry 
            self_symmetric_feature_points_id.append(None)
            self_feature_points_id.append(feature_points_id[i])
            self_symmetric_feature_points_tm.append(None)

    return self_symmetric_feature_points_id, self_feature_points_id, self_symmetric_feature_points_tm, self_symmetric_pattern



def compute_symmetric_control_points_positions_p(feature_points_positions,transform_matrix): #n*3
        """
        Given the position of the updated feature points coordinates, we restore 
        the position of the symmetric feature points
        by using the previously comuted transformation matrix
        """
        if config.setting["gpu_enable"]=="on":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = 'cpu'
        
        nb=feature_points_positions.size()[0]
        
        print(feature_points_positions.device)
        print(device)
                    
        homo_positions= torch.cat([feature_points_positions,torch.ones((nb, 1),device=device)], 1) #n*3
               
        #R=torch.from_numpy(transform_matrix).float().to(device) #3*3  force to be float
        R=torch.from_numpy(transform_matrix).to(device)   #todo         
        coord=torch.matmul(R, homo_positions.t()).t()
        
        symmetric_feature_points_positions=coord[:,:2]  #modified to be 2d
        
        return symmetric_feature_points_positions




if __name__ == "__main__":
    from boundary_utils import get_boundary_edges,assembling_info,control_points,get_grouped_boundaries_anticlc, get_corner_points


    ##########
    ## Unit test of symmetry detection & restoration
    ##########  
    """
    # the control points
    pattern_boundaries= [[2, 410, 421, 425, 21, 26],
                         [368, 375, 384, 50, 42, 29],
                         [130, 125, 91, 80, 55],
                         [158, 170, 205, 209, 134],
                         [237, 219, 325, 337, 341, 232],
                         [287, 296, 301, 262, 259, 238]]
    """
    cloth_2d="./meshes/tshirtm.obj"
    cloth_3d="./meshes/tshirtT.obj"
    cloth_2d_3d="./meshes/tshirt.obj"

    pattern_file=cloth_2d
    sr_mesh=trimesh.load(cloth_3d,process=False)
    nb=len(sr_mesh.vertices)

    print(nb)
    mesh=trimesh.load(pattern_file,process=False)
        
    edges=get_boundary_edges(mesh)
    
    pattern_boundaries=get_grouped_boundaries_anticlc(mesh,get_corner_points) #only the corner points
    print(pattern_boundaries)

    #n2v_dic, v2n_dic, joint_points=assembling_info('./dress/meshes/short_dress.obj' ,nb)
    n2v_dic, v2n_dic, joint_points=assembling_info(cloth_2d_3d ,nb)
    pattern_boundaries=control_points(edges,pattern_boundaries,joint_points) 
    
    print(pattern_boundaries)
    
    print("*********************")
    # Compute the symmetry among the contours composed of the feature points
    feature_points_id, symmetric_feature_points_id, symmetric_feature_points_tm, symmetry_reversed=find_inter_symmetry(mesh, pattern_boundaries)
    print(symmetry_reversed)
    print(feature_points_id)
    print(symmetric_feature_points_id)
    print("*******************")
    self_symmetric_feature_points_id, self_feature_points_id,self_symmetric_feature_points_tm,self_symmetric_pattern=find_intra_symmetry(mesh,feature_points_id)
                           
    print(self_feature_points_id)  
    print(self_symmetric_feature_points_id)
    print(self_symmetric_pattern)
    verts=mesh.vertices[:,:2]
    
    print("*******************")
    #Given the coordinates of a set of control points,
    #use the transofmration matrix to restore the coordinates of their counterparts and compare.
    feature_points_positions=torch.from_numpy(verts[feature_points_id[0]]).double()
    print(verts[symmetric_feature_points_id[0]])
    print(compute_symmetric_control_points_positions_p(feature_points_positions,symmetric_feature_points_tm[0]))