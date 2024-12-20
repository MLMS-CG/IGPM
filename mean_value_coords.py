#Inspired by: https://pages.cs.wisc.edu/~csverma/CS777/bary.html


import numpy as np
from boundary_utils import get_boundary_edges
from green import  bound_coordinates, get_new_bounds


def mean_value_coordinates(cageCoords,queryCoord):
    """
    Args :
        cageCoords :  Coordinates of closed polygon in the Counter
                        clockwise direction. 
        queryCoord:   the xyCoords of the query Point
    
    Returns:
        baryCoords: baryCentric Coords of the query Point.
    
    Reference: Mean Value Coordinates for Arbitrary Planar Polygons:
                Kai Hormann and Michael Floater;
    """
    vj1s=cageCoords
    ss=vj1s-queryCoord
    
    # Page #12, from the paper
    s_ip=np.roll(ss,-1, axis=0)
    
    ris=np.sqrt(np.sum(ss * ss, axis=1))
    rps=np.sqrt(np.sum(s_ip * s_ip, axis=1))
    
    
    Ais= 0.5* (ss[:,0]*s_ip[:,1]-ss[:,1]*s_ip[:,0])
    Dis= np.sum(s_ip * ss, axis=1)
    tanalphas = (ris * rps - Dis) / (2.0 * Ais)
    
    # Equation #11, from the paper
    tanalphas_im=np.roll(tanalphas, 1, axis=0)
    wis=2.0*(tanalphas + tanalphas_im)/ris
    wsum=np.sum(wis)
    baryCoords=wis
    
    
    if(np.abs(wsum)>0):
        baryCoords/wsum

    return baryCoords


def MeanValueCoordinates(queryCoords,cageCoords):
    queries_phis=[]
    
    for i in range(len(queryCoords)):
        queryCoord = queryCoords[i]
        phis=mean_value_coordinates(cageCoords, queryCoord)
        queries_phis.append(phis)
    
    queries_phis=np.stack(queries_phis)
    return queries_phis    
    
def setMVBaryInterpolation(newcageCoords,baryCoords):
    queries_phis=baryCoords
    interior_coords=np.matmul(queries_phis,newcageCoords)/np.sum(queries_phis, axis=1, keepdims=True)
    return interior_coords



def MV_deformation_part(part, global_indices_part, global_con_indices_part, new_verts):
    """ apply the Mean Value Coordiates deformation to the given mesh
     

    Args:
        part: the panel mesh should be deformed, as Trimesh instance
        global_indices_part: the global indices (in the pattern mesh) of all vertices in current panel 
        global_con_indices_part: the global indices of control points in this panel 
        
        new_verts: new vertices containing the updated ctrl points       

    Returns:
        out: deformed vertices of the panel via mean value coordinates

    """
    verts=part.vertices[:,:-1]
    deformed_verts=np.zeros_like(verts)
    
    num_verts=len(verts)
    part_edges=get_boundary_edges(part) 
    
    local_con_indices=[global_indices_part.index(x) for x in global_con_indices_part]
    
    new_ctrls=new_verts[global_con_indices_part][:,:-1]
    
    # get the boundary vertices first        
    indices_segs, alphas_segs, betas_segs = bound_coordinates(local_con_indices, part_edges, verts)
    bound_verts_indices, new_bound_verts=get_new_bounds(new_ctrls, local_con_indices, indices_segs, alphas_segs, betas_segs)
    
    # green coords must be clock-wise 
    bound_verts_indices.reverse()
    new_bound_verts=np.flipud(new_bound_verts)
    
    # get the index of interior vertices
    all_verts_indices=[i for i in range(num_verts)]
    interior_verts_indices=[]
    
    for idx in all_verts_indices:
        if idx not in bound_verts_indices:
            interior_verts_indices.append(idx)
    
    old_bound_verts = verts[bound_verts_indices]
    old_interior_verts = verts[interior_verts_indices]
        
    baryCoords=MeanValueCoordinates(old_interior_verts, old_bound_verts)
       
    new_interior_coords=setMVBaryInterpolation(new_bound_verts,baryCoords)
        
    deformed_verts[bound_verts_indices]=new_bound_verts
    deformed_verts[interior_verts_indices]=new_interior_coords
    
    out=np.concatenate((deformed_verts,part.vertices[:,[-1]]),-1)    
    
    return out  #N*3
