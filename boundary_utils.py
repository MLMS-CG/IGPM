import trimesh
import numpy as np


def get_boundary_indices(mesh):
    unique_edges=mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted,require_count=1)]
    boundary_indices=np.unique(unique_edges.flatten())
    return boundary_indices

def get_boundary_edges(mesh):
    unique_edges=mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted,require_count=1)]
    #print(unique_edges)
    return unique_edges.tolist()

#check if the edge should be apppended to current circle
def chek_add(e,all_):
    if e[0]==all_[-1][1]:
        all_.append(e)

#find out the edges circle starting from the given vertex
def circle(edges,start_pt=26):  
    all_=[]
    start=0
    #determining the starting edge
    for e in edges:        
        if e[0]==start_pt:
            start=e
    
    all_=[start]
    #secure=0
    while (True): # the edges list need be traversed several times before the circle is constructed
        #l1=len(all_)
        for idx, e in enumerate(edges):
            chek_add(e,all_)
        
            if all_[0][0]==all_[-1][1] and len(all_)!=1: #appending next edge
                break
        else:
            continue # no break in the for loop, continue while all over 
        break      
    return all_


def add_edge(prev_v_idx,v_idx,next_v_idx,vertex_visited,pattern_boundaries, pattern_mesh):  
    """
    callback of get_grouped_boundaries_anticlc
    add an edge (vert1,vert2) and mark vert1 as visited 
    """  
    vertex_visited[v_idx] = True
    pattern_boundaries[-1].append([v_idx,next_v_idx])

def get_corner_points(prev_v_idx, v_idx,next_v_idx,vertex_visited,pattern_boundaries, pattern_mesh):
    """
    callback of get_grouped_boundaries_anticlc
    add an the vertx if the angle formed by (v-prev_v , v-next_v) is sharp, and mark as visited 
    """
    vertex_visited[v_idx] = True
    vec1 = pattern_mesh.vertices[v_idx]-pattern_mesh.vertices[prev_v_idx]
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = pattern_mesh.vertices[v_idx]-pattern_mesh.vertices[next_v_idx]
    vec2 = vec2 / np.linalg.norm(vec2)

    if np.linalg.norm(np.dot(vec1,vec2))<0.8:  #<0.45 #Use cos value as the metric, sensitivity can be adjusted
        pattern_boundaries[-1].append(v_idx)

def get_grouped_boundaries_anticlc(pattern_mesh, call_back):
    """
    get either boundary corner points or edges chains, grouped for each panel of the pattern mesh, in anticlock-wise
    """
    # List of boundary edges
    bound_edges=pattern_mesh.edges[trimesh.grouping.group_rows(pattern_mesh.edges_sorted,require_count=1)]
    # List of connected boundary vertices for each boundary vertex
    bound_vert_to_bound_vert = [[-1 for x in range(2)] for y in range(pattern_mesh.vertices.shape[0])]
    
    for edge_idx, edge in enumerate(bound_edges): #only take boundary vertices
        v1=edge[0]
        v2=edge[1]

        bound_vert_to_bound_vert[v1][1] = v2
        bound_vert_to_bound_vert[v2][0] = v1
            
    # Start from the vertex idx 0
    vertex_visited = [False] * pattern_mesh.vertices.shape[0]
    
    v_idx = 0
    pattern_boundaries = []
    # Walk along the pattern boundary
    while v_idx < pattern_mesh.vertices.shape[0]:
        # Skip non-boundary vertices or alreay visited vertices
        if bound_vert_to_bound_vert[v_idx][0] == -1 or vertex_visited[v_idx] == True:
            v_idx = v_idx + 1
            continue
            
        prev_v_idx = -1
        # Walk along the pattern boundary
        pattern_boundaries.append([])
        while True:
            # Find next vertex
            next_v_idx = bound_vert_to_bound_vert[v_idx][1]
            assert(next_v_idx != -1)
            
            # Check if it is a corner vertex
            if prev_v_idx != -1: # what decides is the thing inside, that's the condition 
                call_back(prev_v_idx, v_idx,next_v_idx,vertex_visited,pattern_boundaries,pattern_mesh)    
            # If the next vertex is already visited, the walk is over
            if vertex_visited[next_v_idx]:
                break
            
            # Move to the next vertex
            prev_v_idx = v_idx
            v_idx = next_v_idx
    return pattern_boundaries


def assembling_info(mesh2d3d, nb):
    """
    get the vertices and nodes corresponding relation using 2 dictionaries
    the joint vertices who share same node on which 3 or more panels are sewn together'
    """
    features=[[] for _ in range(nb)] #todo
    
    with open(mesh2d3d,'r') as f1:
        
        lines=f1.readlines()

        for line in lines:
            a=line.replace('\n','').split(" ")
            if a[0]=="f":
                #print(a)
                #print(a[1].split("/")[0],a[1].split("/")[1])
                features[int(a[1].split("/")[0])-1].append(int(a[1].split("/")[1])-1)
                features[int(a[2].split("/")[0])-1].append(int(a[2].split("/")[1])-1)
                features[int(a[3].split("/")[0])-1].append(int(a[3].split("/")[1])-1)
    features=[list(set(f)) for f in features]

    n2v_dic={} # a dic node --> vertices
    for i in range(len(features)):
        n2v_dic[i]=features[i]
    
    join_verts=[]
    dic_v2n={} # a dic vertex ---> node
    #dic_v2n=collections.OrderedDict()
    for node in n2v_dic.keys():
        for vert in n2v_dic[node]:
                dic_v2n[vert]=node

        if len(n2v_dic[node])>2: # node mapping to more than 2 vertices 
            join_verts+=n2v_dic[node]
            
    v2n_dic=dict(sorted(dic_v2n.items(),key=lambda x:x[0]))
            
    return n2v_dic, v2n_dic, join_verts



def insert_joint_points(anticlc_ordered_vertices, pattern_boundaries):
    """
    anticlc_ordered_vertices: the vertices chain staring from the vertex to be added to the control point list
    """
    elements=set(anticlc_ordered_vertices)
    for cps_panel in pattern_boundaries:
        #print("patch")
        #print(patch)
        if set(cps_panel).intersection(elements):  #localise the panel on which the additional vertex is added  
            #print("overlap")
            #print(set(patch).intersection(elements))
            indexing={} # a doc key: idnex of vertex in  boundary vetices chain --> value (index of vertex in corner pts list, index of corner vertex)
            for i,pt in enumerate(cps_panel):
                index=anticlc_ordered_vertices.index(pt) 
                indexing[index]=(i,pt)

            indexing_sorted=dict(sorted(indexing.items(),key=lambda x:x[0])) # sort dic accoring to the location in corner vetices chain

            #print(indexing_sorted)
            
            #as the the corner vetices chain staring from additional joint vertex, 
            #so the insertion position should just be the vertex in panel which has smallest index(right next to index 0) 
            insert_index=indexing_sorted[list(indexing_sorted)[0]][0]

            cps_panel.insert(insert_index, anticlc_ordered_vertices[0])

def control_points(edges,pattern_boundaries,join_points):
    corner_points=[cp for cps in pattern_boundaries for cp in cps ] #corner pts detedcted by 2D angle extended in a single list

    jp_toadd=[]
    for jp in join_points: 
        if jp not in corner_points: #register the join points not yet in the corner pts list 
                jp_toadd.append(jp)

    #insert the joint points to form complete control points list
    for jp in jp_toadd:
        segments=circle(edges,jp)
        chain=segments[0].copy() #Alert: chain is modified later, a copy is needed
        for i in range(1,len(segments)-1):
            chain.append(segments[i][1])
        insert_joint_points(chain,pattern_boundaries) #chain is naturally anticlc ordered
    return pattern_boundaries



if __name__ == "__main__":
    print('unit test')
    
    cloth_2d="./meshes/tshirtm.obj"
    cloth_3d="./meshes/tshirtT.obj"
    cloth_2d_3d="./meshes/tshirt.obj"

    pattern_mesh=trimesh.load(cloth_2d,process=False)
    edges=get_boundary_edges(pattern_mesh)  
    #print(edges)
    
    pattern_boundaries=get_grouped_boundaries_anticlc(pattern_mesh, get_corner_points)
    print(pattern_boundaries)
    
    nodes=trimesh.load(cloth_3d,process=False).vertices
    nb_nodes=len(nodes)
    print(nb_nodes)

    n2v_dic, v2n_dic, join_points=assembling_info(cloth_2d_3d,nb_nodes)
    pattern_boundaries=control_points(edges,pattern_boundaries,join_points)
    print(pattern_boundaries)
    
    