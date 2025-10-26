import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


L=50
#cut = 1.01
cut = np.sqrt(50) + 0.01

torch.set_default_dtype(torch.float64)
#D4 operations
c4 = np.asarray([[0,-1],[1,0]])             #anti-clockwise 90
reflect = np.asarray([[1,0],[0,-1]])        #reflection around line y_id == 0

#[1x,1y,4x,4y,3x,3y,2x,2y] in Yang's note. 
uxy_case1 = torch.tensor([[-1,0,0,1,1,0,0,-1],[0,1,1,0,0,-1,-1,0],[1,0,0,1,-1,0,0,-1],[0,-1,1,0,0,1,-1,0],
                          [1,0,-1,0,1,0,-1,0], [0,-1,0,1,0,-1,0,1],
                          [1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]) 
#for doublet, swap the (Ex,Ey) -> (Ey,Ex)
uxy_case1_2 = torch.tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                           [0,-1,0,1,0,-1,0,1],[1,0,-1,0,1,0,-1,0], 
                           [0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0]]) 

#[1x,4x,3x,2x,1y,4y,3y,2y] in Yang's note.
#uxy_case2 = torch.tensor([[-1,1,1,-1, 1,1,-1,-1],[-1,-1,1,1, -1,1,1,-1],[1,-1,-1,1, 1,1,-1,-1],[1,1,-1,-1, -1,1,1,-1],[0,0,0,0, -1,1,-1,1], [-1,1,-1,1, 0,0,0,0],[1,1,1,1, 0,0,0,0],[0,0,0,0, 1,1,1,1]]) #8*8 matrix
#[1x,1y,4x,4y,3x,3y,2x,2y] in Yang's note. 
uxy_case2 = torch.tensor([[-1,1,1,1,1,-1,-1,-1],[-1,-1,-1,1,1,1,1,-1],[1,1,-1,1,-1,-1,1,-1],[1,-1,1,1,-1,1,-1,-1],
                          [0,-1,0,1,0,-1,0,1], [-1,0,1,0,-1,0,1,0],
                          [1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]) #8*8 matrix
uxy_case2_2 = torch.tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                            [-1,0,1,0,-1,0,1,0],[0,-1,0,1,0,-1,0,1],
                            [0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]])


#[8x,7x,6x,5x,4x,3x,2x,1x,8y,7y,6y,5y,4y,3y,2y,1y]
# uxy_case3 = torch.tensor([[0,-1,1,0,0,1,-1,0,   1,0,0,1,-1,0,0,-1],[1,0,0,-1,-1,0,0,1, 0,-1,-1,0,0,1,1,0],[0,1,1,0,0,-1,-1,0, 1,0,0,-1,-1,0,0,1],[1,0,0,1,-1,0,0,-1, 0,1,-1,0,0,-1,1,0],
#                           [0,1,-1,0,0,-1,1,0, 1,0,0,1,-1,0,0,-1], [1,0,0,-1,-1,0,0,1, 0,1,1,0,0,-1,-1,0],[0,-1,-1,0,0,1,1,0, 1,0,0,-1,-1,0,0,1],[1,0,0,1,-1,0,0,-1, 0,-1,1,0,0,1,-1,0],
#                           [0,0,0,0,0,0,0,0, 0,-1,1,0,0,-1,1,0],[-1,0,0,1,-1,0,0,1, 0,0,0,0,0,0,0,0],[1,0,0,1,1,0,0,1, 0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,1,1,0,0,1,1,0],
#                           [0,0,0,0,0,0,0,0, 1,0,0,-1,1,0,0,-1],[0,1,-1,0,0,1,-1,0, 0,0,0,0,0,0,0,0],[0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1]]) #16*16 matrix

#[8x,8y,7x,7y,6x,6y,5x,5y,4x,4y,3x,3y,2x,2y,1x,1y]
uxy_case3 = torch.tensor([[0,1, -1,0, 1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1], [1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1, 0,1, 1,0],[0,1, 1,0, 1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1],[1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1, 0,1, -1,0],
                          [0,1,1,0,-1,0,0,1, 0,-1, -1,0, 1,0, 0,-1], [1,0, 0,1, 0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0],[0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0, 1,0, 0,1],[1,0, 0,-1, 0,1, 1,0, -1,0, 0,1, 0,-1, -1,0],
                          [0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],[-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],
                          [1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],[0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],
                          [0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],[0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],
                          [0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0],[0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1]]) #16*16 matrix
uxy_case3_2 = torch.tensor([[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0], [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                          [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0], [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                          [-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],[0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],
                          [0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],[1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],
                          [0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],[0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],
                          [0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1], [0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0]])  #16*16 matrix
#for doublet, swap the (Ex,Ey) -> (Ey,Ex)







uxy_case1_ref_choose = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0]])
uxy_case1_ref_choose_2 = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0]])




uxy_case2_ref_choose = uxy_case1_ref_choose
uxy_case2_ref_choose_2 = uxy_case1_ref_choose_2

uxy_case3_ref_choose = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0]])
uxy_case3_ref_choose_2 = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0]]) #Swap doublet



# print("Sanity check, expected 8x8 torch matrix:", uxy_case1.shape)
# print("Sanity check, expected 8x8 torch matrix:", uxy_case2.shape)
# print("Sanity check, expected 16x16 torch matrix:", uxy_case3.shape)



########## util function
def sort_two(x1,x2):
    return tuple([min(x1,x2),max(x1,x2)])


def set_posid(pxy, width):
    return int(pxy[0]%width+(pxy[1]%width)*width)

def set_posxy(pos, width):
    return np.asarray([int(pos%width), int(pos//width)]) #compute x,y. pos is flattened array. So This code makes the 1D array as 2D.

def set_shift(posxy, prxy, height, width): # translating the idx number considering boundary condition
    [x,y] = prxy - posxy
    if x > width//2:
        x = - (width-x)
    if x < -width//2:
        x = x%width
    if y > height//2:
        y = - (height-y)
    if y < -height//2:
        y = y%height
    return np.asarray([int(x), int(y)])

################ find id and correspond matrix

def find_neighbor_id(pos, cut_rad, height, width): #imagine little arrow keep rotating.
    posxy = set_posxy(pos, width)
    #print(posxy)
    out = np.asarray([2*pos])
    out = np.append(out, out+1) # To make (ux[pos], uy[pos]) pairing
    ref_point = []
    cut_int = int(np.ceil(cut_rad))
    for yi in range(cut_int):
        xi_start = yi or 1
        for xi in range(xi_start, cut_int):
            shift = [xi,yi]
            if np.linalg.norm(shift)<=cut_rad:
                if(yi==0):
                    for i in range(4): #total 8 points
                        #print(posxy + shift)
                        out = np.append(out, 2*set_posid(posxy+shift ,width))
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1)
                        shift = np.dot(c4,shift)
                elif (xi==yi):
                    for i in range(4): #total 8 points
                        out = np.append(out, 2*set_posid(posxy+shift ,width))
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1)
                        shift = np.dot(c4,shift)
                else:
                    shift_r = np.dot(c4, np.dot(reflect,shift))
                    for i in range(4): #total 16 points
                        out = np.append(out, 2*set_posid(posxy+shift ,width))
                        ref_point.append(2*set_posid(posxy+shift ,width))
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1)
                        ref_point.append(2*set_posid(posxy+shift ,width)+1)
                        out = np.append(out, 2*set_posid(posxy+shift_r ,width))
                        ref_point.append(2*set_posid(posxy+shift_r ,width))
                        out = np.append(out, 2*set_posid(posxy+shift_r ,width)+1)
                        ref_point.append(2*set_posid(posxy+shift_r ,width)+1)
                        
                        
                        shift = np.dot(c4, shift)
                        shift_r = np.dot(c4, shift_r)
            else:
                break
    #out = np.append(out, out[2:]) #array([pos, pos, {ux_1,...,ux_N}, {uy_1,....,uy_N}])
    
    return torch.tensor(out.astype(np.int16)), torch.tensor(np.array(ref_point).astype(np.int16))
#L = 10
#a,b = find_neighbor_id(1, np.sqrt(5)+0.01,L,L)
#print("This is the index", a.shape)
#print("This is the ref index", b)

def uxy_irrep_reference_matrix(cut_rad, height, width):
    cut_int = int(np.ceil(cut_rad))
    irrep_matrix = torch.tensor([[1,1], [1,-1]])
    irrep_matrix2 = torch.tensor([[0,0], [0,0]])
    reference = torch.ones((1,16))
    reference_to_matrix = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                                     [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]]) # 2x16 matrix; for initial irrep
    reference_to_matrix2 = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    #reference_choose = torch.tensor([[0,0,1,0,0,0,0,0]])
    #reference_to_matrix = torch.tensor([[1]])
    #copy_doublet = torch.tensor([[0]])
    for yi in range(cut_int):
        xi_start = yi or 1
        for xi in range(xi_start, cut_int):
            shift = [xi,yi]
            if np.linalg.norm(shift)<=cut_rad:
                dis_factor = 0.5*(np.cos(np.linalg.norm(shift)*np.pi/(cut_rad+1.0))+1.0)
                if(yi==0):
                    irrep_matrix = torch.block_diag(irrep_matrix, uxy_case1)
                    irrep_matrix2 = torch.block_diag(irrep_matrix2, uxy_case1_2)
                    reference_to_matrix = torch.block_diag(reference_to_matrix, uxy_case1_ref_choose)
                    reference_to_matrix2 = torch.block_diag(reference_to_matrix2, uxy_case1_ref_choose_2)
                    reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                    
                elif (xi==yi):
                    irrep_matrix = torch.block_diag(irrep_matrix, uxy_case2)
                    irrep_matrix2 = torch.block_diag(irrep_matrix2, uxy_case2_2)
                    reference_to_matrix = torch.block_diag(reference_to_matrix, uxy_case2_ref_choose)
                    reference_to_matrix2 = torch.block_diag(reference_to_matrix2, uxy_case2_ref_choose_2)
                    reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                    
                else:
                    irrep_matrix = torch.block_diag(irrep_matrix, uxy_case3)
                    irrep_matrix2 = torch.block_diag(irrep_matrix2, uxy_case3_2)
                    reference_to_matrix = torch.block_diag(reference_to_matrix, uxy_case3_ref_choose)
                    reference_to_matrix2 = torch.block_diag(reference_to_matrix2, uxy_case3_ref_choose_2)
                    reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                   
            else:
                break
                

    return irrep_matrix, irrep_matrix2, reference, reference_to_matrix, reference_to_matrix2#, reference_choose, copy_doublet

#print("irrep matrix shape", uxy_irrep_reference_matrix(np.sqrt(20)+0.01,1,1).shape)



class Descriptors:
    def __init__(self, util_parameters, device, L): #util_parameters = util_matrix.pt -> It has all these dictionary things, device = cpu or gpu(make sure they are in same divice)
        self.neigh_id = util_parameters['neigh_id']
        self.ref_id = util_parameters['ref_id']
        self.uxy_irrep = util_parameters['uxy_irrep']
        self.uxy_irrep_2 = util_parameters['uxy_irrep_2']
        self.uxy_ref = util_parameters['uxy_ref']
        self.uxy_reference_to_matrix = util_parameters['uxy_reference_to_matrix']
        self.uxy_reference_to_matrix_2 = util_parameters['uxy_reference_to_matrix_2']
        self.L = L
        self.device = device

    def set_to_device(self):
        self.neigh_id = self.neigh_id.to(self.device)
        self.ref_id   = self.ref_id.to(self.device)
        self.uxy_irrep = self.uxy_irrep.to(self.device)
        self.uxy_irrep_2 = self.uxy_irrep_2.to(self.device)
        self.uxy_ref = self.uxy_ref.to(self.device)
        self.uxy_reference_to_matrix = self.uxy_reference_to_matrix.to(self.device)
        self.uxy_reference_to_matrix_2 = self.uxy_reference_to_matrix_2.to(self.device)
        
        
    ############ calculate bond and chirality (?)
    def sep_uxy(self, u): #seperate every input data
        #print(self.neigh_id)
        #print(u.shape)
        uxy = u[self.neigh_id]
        #print("center idx",self.neigh_id[180])
        #print("uxy shape before square",uxy.shape)
        #print("uxy center",uxy[180])
        uxy = torch.cat((uxy[:,:2]**2, uxy[:,2:]), dim=1)

        
        
    
        return uxy

    ############ calculate descriptor
    def descriptor_uxy(self, u, uxy_config):
        uxy_case3 = torch.tensor([[0,1, -1,0, 1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1], [1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1, 0,1, 1,0],[0,1, 1,0, 1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1],[1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1, 0,1, -1,0],
                          [0,1,1,0,-1,0,0,1, 0,-1, -1,0, 1,0, 0,-1], [1,0, 0,1, 0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0],[0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0, 1,0, 0,1],[1,0, 0,-1, 0,1, 1,0, -1,0, 0,1, 0,-1, -1,0],
                          [0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],[-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],[1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],[0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],
                          [0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],[0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],[0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0],[0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1]], dtype=torch.float64)
        
        #print(self.uxy_irrep.shape)
        #print(uxy_config.t().shape)
        #print(uxy_config.t()[:,180])
        irrep = torch.matmul(self.uxy_irrep, uxy_config.t()).t()
        irrep_2 = torch.matmul(self.uxy_irrep_2, uxy_config.t()).t()
        #print("irrep shape", irrep.shape)
        #print(self.ref_id.shape)
        uxy_value_ref = u[self.ref_id]
        #print("ref_id shape", self.ref_id.shape)
        #print("uxy_ref.shape",uxy_ref.shape)
        uxy_ref_avg = torch.zeros((self.L*self.L,16))
        
        
        #print(uxy_ref[:,0::16].sum(dim=1))
        for start in range(16):
            uxy_ref_avg[:,start] += uxy_value_ref[:,start::16].sum(dim=1)
            #print(start)
            
        #uxy_ref_avg2 = uxy_value_ref.view(self.L*self.L,-1,16).sum(dim=1)
        #print("difference", uxy_ref_avg2-uxy_ref_avg.sum())
        irrep_ref = torch.matmul(uxy_case3, uxy_ref_avg.t()).t()
        #print("irrep_ref",irrep_ref.shape)
        
        length = self.uxy_ref.shape[0]
        #print(length)
        #uxy_ref_avg_trans = uxy_ref_avg.t()
        irrep_ref_usage = irrep_ref.repeat(1, length)
        #print("irrep_ref_usage shape", irrep_ref_usage.shape)
        #print("reference matrix", self.uxy_reference_to_matrix.shape)
        ref_irrep = torch.matmul(self.uxy_reference_to_matrix, irrep_ref_usage.t()).t()
        ref_irrep_2 = torch.matmul(self.uxy_reference_to_matrix_2, irrep_ref_usage.t()).t()
        #print("ref_irrep", ref_irrep.shape)
        #print("hi",uxy_ref_avg.shape)
        # (10x10) x (16x10).t -> (10x16).t() --> (16x10)
        
        
        #print("This is uxy_irrep data shape, guess ", self.uxy_irrep.shape)
        #print("This is Qx config data shape, guess ", uxy_config.shape)
        #print("This is irrep shape", irrep.shape)
        # reference_neigh = torch.matmul(Qx_config, self.Qx_ref)
        # print("This is Qx_ref shape", self.Qx_ref.shape)
        # print("This is reference_neigh shape", reference_neigh.shape)
        # reference_irrep = torch.matmul(Qx_case3.double().to(self.device), reference_neigh.t()).t()
        # print("This is Qx_case3 shape", Qx_case3.shape)

        # reference irrepresentation sign
        # reference_irrep_sign = reference_irrep.detach().clone()
        # reference_irrep_sign[reference_irrep_sign>=0] = 1
        # reference_irrep_sign[reference_irrep_sign<0] = -1
        # reference_irrep[:,:4] = reference_irrep_sign[:,:4]
        # reference_irrep[:,0] = 1

        # choose coorespond reference for irrep
        #reference_irrep = torch.matmul(self.Qx_ref_choose, reference_irrep.t()).t()
        # calculate reference matrix
        #reference_mul_irrep = reference_irrep * irrep
        #irrep_mul_irrep = irrep * irrep
        irrep_mul_ref_irrep_1 = irrep * ref_irrep
        irrep_mul_ref_irrep_2 = irrep_2 * ref_irrep_2
        #ref_irrep_mat = self.Qx_ref_to_mat.expand(irrep.shape[0], irrep.shape[1], irrep.shape[1])
        #irrep_mat = self.Qx_copy_doublet.expand(irrep.shape[0], irrep.shape[1], irrep.shape[1])
        #first = torch.matmul(ref_irrep_mat, reference_mul_irrep.view(-1,irrep.shape[1],1))
        #second = torch.matmul(irrep_mat, irrep_mul_irrep.view(-1,irrep.shape[1],1))

        #input_descriptor = first + second
        #return input_descriptor.view(-1,input_descriptor.shape[1])
        return irrep_mul_ref_irrep_1+irrep_mul_ref_irrep_2
    
    # def descriptor_Qz(self, Qz_config):
    #     irrep = torch.matmul(self.Qz_irrep, Qz_config.t()).t()
    #     reference_neigh = torch.matmul(Qz_config, self.Qz_ref)
    #     reference_irrep = torch.matmul(Qz_case3.double().to(self.device), reference_neigh.t()).t()

    #     # reference irrepresentation sign
    #     reference_irrep_sign = reference_irrep.detach().clone()
    #     reference_irrep_sign[reference_irrep_sign>=0] = 1
    #     reference_irrep_sign[reference_irrep_sign<0] = -1
    #     reference_irrep[:,:4] = reference_irrep_sign[:,:4]
    #     reference_irrep[:,0] = 1

    #     # choose coorespond reference for irrep
    #     reference_irrep = torch.matmul(self.Qz_ref_choose, reference_irrep.t()).t()
    #     # calculate reference matrix
    #     reference_mul_irrep = reference_irrep * irrep
    #     irrep_mul_irrep = irrep * irrep
    #     ref_irrep_mat = self.Qz_ref_to_mat.expand(irrep.shape[0], irrep.shape[1], irrep.shape[1])
    #     irrep_mat = self.Qz_copy_doublet.expand(irrep.shape[0], irrep.shape[1], irrep.shape[1])
    #     first = torch.matmul(ref_irrep_mat, reference_mul_irrep.view(-1,irrep.shape[1],1))
    #     second = torch.matmul(irrep_mat, irrep_mul_irrep.view(-1,irrep.shape[1],1))

    #     input_descriptor = first + second
    #     print("This is input descriptor shape",input_descriptor.view(-1,input_descriptor.shape[1]).shape)
    #     return input_descriptor.view(-1,input_descriptor.shape[1])


     ##### calculate all descriptors

    def cal_descriptor(self, u):
        #print("1")
        uxy= self.sep_uxy(u)
        des_uxy = self.descriptor_uxy(u, uxy)
        #des_Qx = self.descriptor_Qx(Qx)
        #des_Breath = self.descriptor_Qz(Breath)
        return des_uxy


if __name__ == "__main__":
    neigh_list_shape = find_neighbor_id(0, cut, L, L)[0].shape
    ref_idx_shape = find_neighbor_id(0, cut, L, L)[1].shape
    #print("In main, ref_idx_shape",ref_idx_shape)
    #print(neigh_list_shape)
    neigh_id = torch.zeros(L*L,neigh_list_shape[0])
    ref_id = torch.zeros(L*L, ref_idx_shape[0])
    for i in range(L*L):
        neigh_id[i,:] = find_neighbor_id(i, cut, L, L)[0]
        ref_id[i,:] = find_neighbor_id(i, cut, L, L)[1]
        #print(find_neighbor_id(i, cut, L, L)[1])
    #print("Shape of neighbor id!: ", neigh_id.shape)
    
    
    uxy_irrep, uxy_irrep_2, uxy_reference, uxy_reference_to_matrix, uxy_reference_to_matrix_2 = uxy_irrep_reference_matrix(cut, L, L)
    uxy_irrep = uxy_irrep.to(dtype=torch.float64)
    uxy_irrep_2 = uxy_irrep_2.to(dtype=torch.float64)
    uxy_reference = uxy_reference.to(dtype=torch.float64)
    uxy_reference_to_matrix = uxy_reference_to_matrix.to(dtype=torch.float64)
    uxy_reference_to_matrix_2 = uxy_reference_to_matrix_2.to(dtype=torch.float64)
    #print("uxy_irrep shape : ",uxy_irrep.shape)
    
    #ueg = torch.ones(L*L, dtype=torch.float32)
    #ueg = torch.rand(L*L, dtype=torch.float32)
    #print("ueg shape", ueg.shape)
    #uxy_neighbor = ueg[neigh_id.long()]
    #print(uxy_neighbor)
    
    #print("ueg.t shape", ueg.t().shape)
    #print("uxy_neighbor shape", uxy_neighbor.shape)
    
    #irrep = torch.matmul(uxy_irrep, uxy_neighbor.t()).t()
    #fvs = irrep * irrep
    #print(fvs.shape)
    #print("Here", Qx_irrep)
    #Qz_irrep, Qz_ref, Qz_ref_to_mat, Qz_ref_choose, Qz_copy_doublet = Qz_irrep_reference_matrix(cut, L, L)
    base_save = '/Users/jangho/Downloads/ML_peierls/util/'
    
    torch.save({"neigh_id": neigh_id.long(),"ref_id": ref_id.long(),"uxy_irrep": uxy_irrep.double(), "uxy_irrep_2":uxy_irrep_2.double(), "uxy_ref":uxy_reference.double(), "uxy_reference_to_matrix":uxy_reference_to_matrix.double(), "uxy_reference_to_matrix_2":uxy_reference_to_matrix_2.double()},base_save+'util_matrix.pt')
