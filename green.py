#Inspired by: https://pages.cs.wisc.edu/~csverma/CS777/bary.html

import numpy as np
from boundary_utils import get_boundary_edges
from seaming_line import find_chains
      
def bound_coordinates(con_indices, edges, verts):
    
    """ For each ""segment" (a pair of two control points) ,compute the relative position of 
    the bound vertices by refering to its neighbouring control points, the positions are 
    reresented as "coordinates" (alpha, beta)as follows:
        
        v = c_st + alpha x_vec + betas R90 x_vec, where x_vec= c_end- c_st
        
        
    therefore vy using previous verts positions
    
    alphs=dot(v-c_st,x) / dot(x,x)
    beta= dot(v-c_st,y) / dot(y,y)

    Args:
        con_indices: local control point indices
        edges: boundary edges list 
        verts: vertices in arr

    Returns:
        inseg_indices_segs
        alphas_segs
        betas_segs

    """
    
    
    num_con=len(con_indices)
    
    inseg_indices_segs=[]
    alphas_segs=[]
    betas_segs =[]
    
    for i in range(num_con):
        
        seg_cs=con_indices[i]
        seg_ce=con_indices[(i+1)%num_con]
        
        #get all the vertices between 2 ctrl points(included) in order
        links, _=find_chains([seg_cs,seg_ce],edges)
        links=np.array(links).flatten()
        _, idx = np.unique(links, return_index=True)
        seg_points=links[np.sort(idx)]
        
        inseg_indices=seg_points[1:-1].tolist() # only the vertices connecting two ctrl points
        
        # axies
        x_vec=verts[seg_ce]-verts[seg_cs]
        y_vec=np.matmul(np.array([[0,-1],[1,0]]),x_vec)
        
        
        alphas=np.sum((verts[inseg_indices]-verts[seg_cs]) * x_vec, axis=1) / np.sum(x_vec *x_vec) 
        betas =np.sum((verts[inseg_indices]-verts[seg_cs]) * y_vec, axis=1) / np.sum(y_vec *y_vec)
        
        inseg_indices_segs.append(inseg_indices)
        alphas_segs.append(alphas)
        betas_segs.append(betas)
        
    return inseg_indices_segs, alphas_segs, betas_segs
    
def get_new_bounds(new_ctrls, local_con_indices, indices_segs, alphas_segs, betas_segs):
    """ For each ""segment" (a pair of two control points) ,compute the new positions of 
    the bound vertices by using precomputed "coordinates" as follows:
        
        v' = c_st' + alpha x_vec' + betas R90 x_vec', where x_vec'= c_end'- c_st'
        
    Args:
        new_ctrls: new control points positions
       local_con_indices: local control point indices
       indices_segs: bound vertices indices between 2 control points per segment 
       alphas_segs: alphas per segment 
       betas_segs: betas per segment


    Returns:
        indices: all
        bound_verts: all 

    """
    indices=[]
    new_verts = []
    num_con=len(new_ctrls)
    for i in range(num_con):
        
        # index of the starting control point &  the indices of the dollowing bound vertices
        indices+=([local_con_indices[i]]+indices_segs[i])
        
    
        vert_seg_cs=new_ctrls[i]
        vert_seg_ce=new_ctrls[(i+1)%num_con]
        
        x_vec=vert_seg_ce-vert_seg_cs
        y_vec=np.matmul(np.array([[0,-1],[1,0]]),x_vec)
        
                
        # interpolating bound vertices between 2 ctrl points   
        alphas=alphas_segs[i].reshape(-1,1) # prepare(n,1) for broadcast with (2,)
        betas=betas_segs[i].reshape(-1,1)
        inseg_points=vert_seg_cs + alphas* x_vec + betas* y_vec
                
        # the starting control point & the dollowing bound vertices
        new_verts.append(vert_seg_cs.reshape(-1,2))
        new_verts.append(inseg_points)   
    
    bound_verts=np.concatenate(new_verts, axis=0)
    return indices, bound_verts

def green_coordinates(cageCoords,queryCoord):
    
    """
    // Input :
    //      1. cageCoords : Coordinates of closed polygon in the
    //                      Clockwise direction. The input is not tested inside.
    //      2. queryCoord:  the xyCoords of the query Point
    // Output:
    //       1:  baryCoords: baryCentric Coords of the query Point with respect
    //           to cageCoords.
    //
    // Reference: Green Coordinates
    //            Yaron Lipman, David Levin and Daniel Cohen-or
    ///////////////////////////////////////////////////////////////////////////
    """
    distances=np.linalg.norm(cageCoords-queryCoord,axis=1)**2
    if (distances<= 1.0E-20).any():
        print(" TOO ClOSED ")
        
    vj1s=cageCoords
    vj2s=np.roll(cageCoords, -1, axis=0) #roll array elements inversely along 0 axis.
    
    a_vecs=vj2s-vj1s
    njs = a_vecs[:,[1,0]]
    njs[:,1] = -njs[:,1]
    
    b_vecs=vj1s-queryCoord
    
    Qs=np.sum(a_vecs * a_vecs, axis=1)
    Ss=np.sum(b_vecs * b_vecs, axis=1)
    Rs=2 * np.sum(a_vecs * b_vecs, axis=1)
    BAs=np.sum(b_vecs * njs, axis=1)  # do not normalize Normal vector here.
    Vs=4 * Ss * Qs - Rs*Rs
    assert((Vs > 0.0).all())
    SRTs=np.sqrt(Vs)
    assert((SRTs > 0.0).all())
    
    L0s = np.log(Ss)
    L1s = np.log(Ss + Qs + Rs)

    A0s = np.arctan(Rs/SRTs) / SRTs
    A1s = np.arctan((2*Qs+Rs) / SRTs) / SRTs

    A10s = A1s - A0s
    L10s = L1s - L0s
    
    psis =  np.sqrt(Qs) / (4.0 * np.pi) * ((4.0 * Ss - (Rs * Rs / Qs)) * A10s + (Rs / (2.0 * Qs)) * L10s + L1s - 2)
    
    
    phis = (BAs / (2.0 * np.pi)) * ((L10s / (2.0 * Qs)) - A10s * (2.0 + Rs / Qs))
    phis -= np.roll( (BAs / (2.0 * np.pi)) * ((L10s / (2.0 * Qs)) - A10s * Rs / Qs), 1,  axis=0)  #before it is 1,2-->0, so roll it to 0,1-->n-1

    
    #psis/=np.sum(psis)
    phis/=np.sum(phis)
    
    return psis, phis

def GreenCoordinates(queryCoords, cageCoords):
    """ 
    For each queryCoord, get the green coords

    """
    queries_psis=[]
    queries_phis=[]
    
    for i in range(len(queryCoords)):
        queryCoord = queryCoords[i]
        psis, phis=green_coordinates(cageCoords, queryCoord)
        
        queries_psis.append(psis)
        queries_phis.append(phis)
    
    queries_psis=np.stack(queries_psis)
    queries_phis=np.stack(queries_phis)
    
    baryCoords=(queries_psis, queries_phis)
    return baryCoords

def setGreenBaryInterpolation(newcageCoords,newSegLens,orgSegLens,newcageNormals,baryCoords):
    sis=newSegLens/orgSegLens
    
    queries_psis=baryCoords[0]
    queries_phis=baryCoords[1]
    interior_coords=np.matmul(queries_phis,newcageCoords)+ np.matmul(queries_psis,sis*newcageNormals)
    return interior_coords

def getSegLens(cageCoords):
    vj1s=cageCoords
    vj2s=np.roll(cageCoords, -1, axis=0) #roll array elements inversely along 0 axis.
    
    a_vecs=vj2s-vj1s
    segLens=np.linalg.norm(a_vecs,axis=1)[:, None] # seg lens
    
    return segLens

def getCageNormals(cageCoords):
    
    vj1s=cageCoords
    vj2s=np.roll(cageCoords, -1, axis=0) #roll array elements inversely along 0 axis.
    
    a_vecs=vj2s-vj1s
    njs = a_vecs[:,[1,0]]
    njs[:,1] = -njs[:,1]
     
    normals=njs/np.linalg.norm(njs,axis=1)[:, None] # normalized normals
    
    return normals

def GBC_deformation_part(part,global_indices_part,global_con_indices_part, new_verts):
    """ apply the modified laplacian deformation to the given mesh
     

    Args:
        part: the panel mesh should be deformed, as Trimesh instance
        global_indices_part: the global indices (in the pattern) of all vertices in current panel 
        global_con_indices_part: the global indices of control points in this panel 
        
        new_verts: global vertices with expected ctrl points           

    Returns:
        out: deformed vertices of the panel via green coordinates

    """
    verts=part.vertices[:,:-1]
    deformed_verts=np.zeros_like(verts)
    
    num_verts=len(verts)
    part_edges=get_boundary_edges(part) 
    
    local_con_indices=[global_indices_part.index(x) for x in global_con_indices_part]
    
    new_ctrls=new_verts[global_con_indices_part][:,:-1]
    
    #new_ctrls=vertices[global_con_indices_part]
        
    indices_segs, alphas_segs, betas_segs = bound_coordinates(local_con_indices, part_edges, verts)
    bound_verts_indices, new_bound_verts=get_new_bounds(new_ctrls, local_con_indices, indices_segs, alphas_segs, betas_segs)
    
    # green coords must be clock-wise 
    bound_verts_indices.reverse()
    new_bound_verts=np.flipud(new_bound_verts)
    
    
    all_verts_indices=[i for i in range(num_verts)]
    interior_verts_indices=[]
    
    for idx in all_verts_indices:
        if idx not in bound_verts_indices:
            interior_verts_indices.append(idx)
    
    
    old_bound_verts = verts[bound_verts_indices]
    old_interior_verts = verts[interior_verts_indices]
        
    baryCoords=GreenCoordinates(old_interior_verts, old_bound_verts)
   
    # prepare for the interpolation
    newSegLens=getSegLens(new_bound_verts)
    orgSegLens=getSegLens(old_bound_verts)
    
    newcageNormals=getCageNormals(new_bound_verts)
    
    new_interior_coords=setGreenBaryInterpolation(new_bound_verts,newSegLens,orgSegLens,newcageNormals,baryCoords)
    
    
    deformed_verts[bound_verts_indices]=new_bound_verts
    deformed_verts[interior_verts_indices]=new_interior_coords
    
    out=np.concatenate((deformed_verts,part.vertices[:,[-1]]),-1)    
    
    return out  #N*3


