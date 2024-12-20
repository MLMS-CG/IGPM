from trimesh.geometry import faces_to_edges
import numpy as np
from trimesh import grouping
from scipy.sparse import csc_matrix

def get_adjvs(mesh):
    """Fetches neighboring vertices for all the vertices in a mesh.

    Retrieves neighboring vertices in np.array 

    Args:
        mesh: A Trimesh instance.


    Returns:
        A list of grouped neighboring vertices, each element is an array  
        
        [ arr([1,2,3,4]),  #(of vert 0)  
          arr([0,5,6]),    #(of vert 1 )
          ..
        ]

    """
    
    # edges are anticlc, so interior ones are alreay repeated with direction
    edges, edges_face = faces_to_edges(mesh.faces, return_index=True) 
    old_edges=edges.copy()
    edges.sort(axis=1) # sort along axis=1, inter edges twice bound once
    
    bound_edges=old_edges[grouping.group_rows(edges,require_count=1)] 
    new_edges=np.vstack((old_edges,bound_edges[:, [1, 0]])) # overturn bound edges and add to have mutual pointing
    sorted_edges=new_edges[new_edges[:, 0].argsort()]
    
    #Group neighboring vertices in ordered edges which have equal starting vetices(first column)
    list_adjvs=np.split(sorted_edges[:,1], np.unique(sorted_edges[:,0], return_index=1)[1][1:],axis=0)
    return list_adjvs

def create_csc(asub, m, n):
    """ 
    create sparse matrix by equiping non-zero elements with identity matrix(2d)

    Args:
        asub: A dense representaion as np.array instance.
        m,n : size of the asub


    Returns:
        sp_A: sparse matrix corresponding to asub

    """
    row=[]
    col=[]
    data=[]
    
    for i in range(m):
        for j in range(n):
            row+=[i*2,i*2+1]
            col+=[j*2,j*2+1]
            value=asub[i][j]
            data+=[value,value]

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    sp_A = csc_matrix((data, (row, col)), shape=(m*2, n*2), dtype=float)
    
    return sp_A
