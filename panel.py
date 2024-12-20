import numpy as np
from boundary_utils import get_boundary_edges
from seaming_line import find_chains
import torch
from modified_laplacian import get_adjvs, create_csc
import config


if config.setting["gpu_enable"]=="on":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'


class Panel:
    def __init__(self, part, global_part_indice, global_con_indice, ctrl_reverse, mode):
        
        
        self.ctrl_reverse=ctrl_reverse
        
        self.inter_verts_indices=[]
        self.bound_verts_indices=[]
        
        #for local coordinates system by two control points 
        self.inseg_indices_segs=[]
        self.alphas_segs=[]
        self.betas_segs =[]
        
        
        self.part=part
        self.verts=part.vertices[:,:-1] # n,3-->n,2
        self.x_old = torch.from_numpy(self.verts).float().reshape(-1).to(device)

        if self.ctrl_reverse:
            global_con_indice.reverse()
        
        self.global_part_indices=global_part_indice # for filling the deformed pattern mesh
        self.local_con_indices=[global_part_indice.index(global_con) for global_con in global_con_indice] #boundary indices but in local
        
        self.nb_con=len(self.local_con_indices)
        
        self.__get_inter_bound_indices()  # prepare for bound computaion	
        self.mean_value_prepare() #by default is MVC, most appropriate one according to experiments
        
        self.mode=mode	#TODO: to support mode switch
        if self.mode: 
            #print(self.bound_verts_indices)	
            #self.green_prepare()	
            list_adjvs=get_adjvs(self.part)            	
            self.A,self.B,self.C = self.__get_laplacian_matrices(list_adjvs) # interior verrtices computaion uing Laplacian editing

    def __bound_coordinates(self, edges):
        """
        build the local coordinates system by using the vector connecting two control points 
        after normalization, and another axis by rotating it 90 degree
        refer to the Equation 3.1 in the thesis. 
        """
        for i in range(self.nb_con):
            
            seg_cs=self.local_con_indices[i]
            seg_ce=self.local_con_indices[(i+1)%self.nb_con]
            
            #print(seg_cs,seg_ce)
            
            
            #get all the vertices between 2 ctrl points(included) in order
            links, _=find_chains([seg_cs,seg_ce],edges)
            #print(links)
            links=np.array(links).flatten()
            _, idx = np.unique(links, return_index=True)
            seg_points=links[np.sort(idx)]
            
            inseg_indices=seg_points[1:-1].tolist() # only the vertices connecting two ctrl points
            
            # axies
            x_vec=self.verts[seg_ce]- self.verts[seg_cs]
            y_vec=np.matmul(np.array([[0,-1],[1,0]]),x_vec)
            
            
            alphas=np.sum((self.verts[inseg_indices]-self.verts[seg_cs]) * x_vec, axis=1) / np.sum(x_vec *x_vec) 
            betas =np.sum((self.verts[inseg_indices]-self.verts[seg_cs]) * y_vec, axis=1) / np.sum(y_vec *y_vec)
            
            
            #alphas_tsr=torch.from_numpy(alphas).float().to(device)
            #betas_tsr=torch.from_numpy(betas).float().to(device) #todo
            alphas_tsr=torch.from_numpy(alphas).to(device)
            betas_tsr=torch.from_numpy(betas).to(device)
            
            
            self.inseg_indices_segs.append(inseg_indices)
            self.alphas_segs.append(alphas_tsr) 
            self.betas_segs.append(betas_tsr)   
        
            
    def __get_inter_bound_indices(self):  
        """
        get the indices for both interior & boundary vertices 
        """
        num_verts=len(self.verts)
        part_edges=get_boundary_edges(self.part) 
        
        # interpolating the bound vertices which lay between control points and link all the bound together
        self.__bound_coordinates(part_edges)
        
        for i in range(self.nb_con):      
            # index of the starting control point &  the indices of the following bound vertices
            self.bound_verts_indices+=([self.local_con_indices[i]]+self.inseg_indices_segs[i])
        
        # get the index of interior vertices
        all_verts_indices=[i for i in range(num_verts)]
        
        for idx in all_verts_indices:
            if idx not in self.bound_verts_indices:
                self.inter_verts_indices.append(idx)    
    
    
        
    def __get_laplacian_matrices (self, list_adjvs):
        """ constrcut the modified laplacian deformation matrices and solve the
        linear system.
        
        modified laplacaian cost func: || I*x_inter+B*x_bound-L*x_old ||^2
        
        by gradient=0:
            
            I.T*I*x_inter=I.T*L*x_old-I.T*B*x_bound
            
            A=I.T*I
            B=I.T*L
            C=I.T*B
    
        solve to get the interior vertices     
        """
        n_all = len(self.verts)
        n_inter = len(self.inter_verts_indices) 
        n_bound = len(self.bound_verts_indices)
    
        
        L=np.zeros((n_all,n_all))
        I=np.zeros((n_all,n_inter))
        B=np.zeros((n_all,n_bound))
        
        #laplacian weights   
        for i in range(n_all):
            adjvs=list_adjvs[i]  # adjacent vertices of i-th vertex
            nb=len(adjvs)
    
            L[i][i]=-1
            if i in self.bound_verts_indices:
                j=self.bound_verts_indices.index(i)
                B[i][j]=-1
                
            if i in self.inter_verts_indices:
                j=self.inter_verts_indices.index(i)
                I[i][j]=-1
                
            for n in adjvs:
                L[i][n]=1/nb            
                if n in self.bound_verts_indices:
                    j=self.bound_verts_indices.index(n)
                    B[i][j]=1/nb
                    
                if n in self.inter_verts_indices:
                    j=self.inter_verts_indices.index(n)
                    I[i][j]=1/nb
    
        ItI=np.matmul(I.T,I)
        ItB=np.matmul(I.T,B)
        ItL=np.matmul(I.T,L)
            
        a = create_csc(ItI,n_inter,n_inter).toarray()
        b = create_csc(ItL,n_inter,n_all).toarray()
        c = create_csc(ItB,n_inter,n_bound).toarray()  # sparse to dense
        
        A=torch.from_numpy(a).float().to(device)   # to tensor
        B=torch.from_numpy(b).float().to(device)
        C=torch.from_numpy(c).float().to(device)
        
        return A,B,C
    
    def get_new_bounds(self, new_ctrls):
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
            bound_verts: all 
    
        """
        new_bound_verts = []
        for i in range(self.nb_con):
            
            # index of the starting control point &  the indices of the following bound vertices
            
            vert_seg_cs=new_ctrls[i]
            vert_seg_ce=new_ctrls[(i+1)%self.nb_con]
            
            x_vec=vert_seg_ce-vert_seg_cs
            R=torch.tensor([[0., -1.],[1., 0.]]).to(device).double() #todo
            
            #print(R.dtype)
            #print(x_vec.dtype)
            
            y_vec=torch.matmul(R,x_vec)
            
                    
            # interpolating bound vertices between 2 ctrl points   
            alphas=self.alphas_segs[i].reshape(-1,1) # prepare(n,1) for broadcast with (2,)
            betas=self.betas_segs[i].reshape(-1,1)
            
            inseg_points=vert_seg_cs + alphas* x_vec + betas* y_vec
                    
            # the starting control point & the following bound vertices
            new_bound_verts.append(vert_seg_cs.reshape(-1,2))
            new_bound_verts.append(inseg_points)   
        
        bound_verts=torch.cat(new_bound_verts, axis=0)  # n,2
        return bound_verts


    def solve_laplacian_linear (self, x_old, x_bound):
        """
        Args:
            x_old: 
            x_bound: new position of bound vertices
        Returns:
            x_inter: interior vertices as orders in inter_verts_indices
        
        """
        A=self.A.unsqueeze(0)  # must be (*,n,n)
        b= torch.matmul(self.B,x_old)-torch.matmul(self.C,x_bound)
        
        x_inter=torch.linalg.solve(A, b).reshape(-1,2)  
        return x_inter
    
    
    
    def modified_laplacian(self, new_ctrls):
        """ apply the modified laplacian deformation to the given mesh
         
    
        Args:            
            new_verts: global vertices with expected ctrl points       
    
        Returns:
            out: deformed vertices of the panel
    
        """
        if self.ctrl_reverse:
            new_ctrls=torch.flip(new_ctrls, dims=(0,))
        
        
        new_bound_verts=self.get_new_bounds(new_ctrls)
        
        
        x_bound=new_bound_verts.reshape(-1) 
        interior_coords=self.solve_laplacian_linear(self.x_old, x_bound)  # return (n,2)
        
        
        deformed_verts=torch.zeros(self.verts.shape).to(device)   # 2D
        # fill the empty arr with bound verts and interpolated interior ones
        deformed_verts[self.bound_verts_indices]=new_bound_verts
        deformed_verts[self.inter_verts_indices]=interior_coords
        
        #out=torch.cat([deformed_verts,torch.zeros((deformed_verts.size()[0], 1),device=device)],1)
        out=deformed_verts
        #print("************for the current part************")
        return out
        
        
    def mean_value_prepare(self):
        """
        # implementation of mean value coordinates
        tutorial: https://pages.cs.wisc.edu/~csverma/CS777/bary.html
        https://www.numerical-tours.com/matlab/meshdeform_4_barycentric/
        """
        from mean_value_coords import MeanValueCoordinates
        
        old_bound_verts = self.verts[self.bound_verts_indices]
        old_interior_verts = self.verts[self.inter_verts_indices]
            
        
        baryCoords=MeanValueCoordinates(old_interior_verts, old_bound_verts)
        #self.queries_phis=torch.from_numpy(baryCoords).float().to(device)
        self.queries_phis=torch.from_numpy(baryCoords).to(device)   #todo
        
    def mean_value_coordinates(self, new_ctrls):
        """ 
        apply the modified mean value coordinates based deformation to the given mesh
        """
        
        #deformed_verts=torch.zeros(self.verts.shape).to(device)   # 2D
        deformed_verts=torch.zeros(self.verts.shape).to(device).double()   # 2D
        
        if self.ctrl_reverse:
            new_ctrls=torch.flip(new_ctrls, dims=(0,))
        
        
        new_bound_verts=self.get_new_bounds(new_ctrls)  #must be anti clock order
        deformed_verts[self.bound_verts_indices]=new_bound_verts
                
        new_interior_coords=torch.matmul(self.queries_phis,new_bound_verts)/torch.sum(self.queries_phis, 1, keepdims=True)
        
        deformed_verts[self.inter_verts_indices]=new_interior_coords
        out=deformed_verts
        
        #out=torch.cat([deformed_verts,torch.zeros((deformed_verts.size()[0], 1),device=device)],1)
        
        return out

    def rbf(self, new_ctrls):	
        	
        if self.ctrl_reverse:	
            new_ctrls=torch.flip(new_ctrls, dims=(0,))	
        	
        verts=self.x_old.reshape(-1,2)
	            	
        landmarks=verts[self.local_con_indices]	
                	
        displacements=new_ctrls-landmarks	
        	
        rbfX = RBFInterpolator(landmarks, displacements[:,0].reshape((-1,1)))	
        rbfY = RBFInterpolator(landmarks,displacements[:,1].reshape((-1,1)) )	
        	
        dispalcex=rbfX.interpolate(verts)	
        dispalcey=rbfY.interpolate(verts)	
        	
        	
        deformed_verts=verts+torch.cat((dispalcex,dispalcey),-1)	
        #out=torch.cat([deformed_verts,torch.zeros((deformed_verts.size()[0], 1),device=device)],1)	
        out=deformed_verts

        return out	
    
    def deform(self, new_landmarks):
        """
        several spatial interpolation choices 
        """        	
        if self.mode=="lap":	
            return self.modified_laplacian(new_landmarks)	
        elif self.mode=="mvc":
            return self.mean_value_coordinates(new_landmarks)
        elif self.mode=="green":
            #TODO: support green coordinates after implementing the regularization loss
            pass	
        else:	
            return self.rbf(new_landmarks)	
        	
class RBFInterpolator():	
    def __init__(self, landmarks, f):	
        super().__init__()	
        self.M = f.shape[0]	
        M=self.M	
        self.F = torch.zeros((M+3,1),device=device)	
        self.P = torch.zeros((M, 2),device=device)	
        G = torch.zeros((M + 3,M + 3),device=device)	
        # copy function values	
        self.F[:M] = f	
        # fill xyz coordinates into P	
        self.P=landmarks	
        	
        # the matrix below is symmetric, so I could save some calculations Hmmm. must be a todo	
        for i in range(M):	
            for j in range(M): # M=86 in our case	
                    vec1=landmarks[i] 	
                    vec2=landmarks[j]	
                    distance_squared=torch.sum(torch.square(vec1 - vec2))	
                    G[i][j] = self.g(distance_squared)	
        #Set last 4 columns of G    	
        G[:M,M:]=torch.cat((torch.ones((M,1),device=device),landmarks),axis=1)	
        #Set last 4 rows of G	
        G[M:,:]=G[:,M:].t()	
        self.Ginv = torch.inverse(G)	
        self.A = torch.matmul(self.Ginv,self.F)	
        	
    	
    #RBF interpolation given a point in 3d       	
    def interpolate(self, verts):	
        	
        N=verts.shape[0]	
        verts_=verts.unsqueeze(1)  #n*1*3	
        feature_points=self.P.unsqueeze(0).repeat(N,1,1) #N*M*3	
        Part1=self.g(torch.square(verts_-feature_points).sum(-1).squeeze(-1))  #N*M	
        	
        CC=torch.cat((Part1,torch.ones((N,1),device=device),verts),-1) #N*(M+4)	
       	
        #displacements=torch.matmul(CC,self.A.double())	
        displacements=torch.matmul(CC,self.A)	
        return displacements	
    	
    def g(self,t_squared):	
        return torch.sqrt(torch.log10(t_squared + 1))
