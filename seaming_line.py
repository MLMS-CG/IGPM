import torch
from boundary_utils import circle
import trimesh
import numpy as np


#retrun list of segments connecting the the vertices given in line
def find_chains(line,edges):
    v1=line[0]
    v2=line[1]
    
    patch_edge_circle=circle(edges,v1) # the boundary edges starting from v1
    for i,e in enumerate(patch_edge_circle):
        if e[0]==v2: #stop at v2 and return the two parts v1-->v2 and the rest v2-->v1
            return patch_edge_circle[:i], patch_edge_circle[i:]
        

####
# seam loss part non-disclosed
#####