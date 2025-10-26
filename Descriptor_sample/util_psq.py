import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


L=66 # N = L * L
#cut = 1.01
cut = np.sqrt(50) + 0.01 # Cut radius (boundary is not included)

torch.set_default_dtype(torch.float32)
#D4 operations


'''
Description of this code (Origianlly coming from Sheng Zhang), changed by Ho Jang(Jason)

Expected Data to machine; Input : displacement (1D N(L*L) array) --> Output : local energy at each site | d(Summation over all site energy)/d(displacement) = - (Force)

Goal of this code : In training code, from this util.py, (displacement) --> F.V. | e.x. x_temp = util_func.cal_descriptor(temp_coordinate) in training code

In Peierls model, displacment has two components(ux, uy). --> Input data is filed as torch.tensor([u[0]x, u[0]y, u[i]x, u[i]y, u[N]x, u[N]y]) | irrep matrix form depends on this arrangement

Feature Variables(F.V.), which will be given to Machine input, has form, F.V. = (irrep) * (ref_irrep).

To accomplish this, I generated "two matrices" for each "irrep" and "ref_irrep". More detail is in the note.


'''


# Type 1
# Peierls model, displacement irrep matrkx
# [1x,1y,4x,4y,3x,3y,2x,2y] in Yang's note. | Ordering is described later.
# 8*8 matrix
uxy_case1 = torch.tensor([[-1,0,0,1,1,0,0,-1],[0,1,1,0,0,-1,-1,0],
                          [1,0,0,1,-1,0,0,-1],[0,-1,1,0,0,1,-1,0],
                          [1,0,-1,0,1,0,-1,0], [0,-1,0,1,0,-1,0,1],
                          [1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]])
 
#for doublet, swap the (Ex,Ey) -> (Ey,Ex) (in the note)
uxy_case1_2 = torch.tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                           [0,-1,0,1,0,-1,0,1],[1,0,-1,0,1,0,-1,0], 
                           [0,1,0,1,0,1,0,1], [1,0,1,0,1,0,1,0]]) 


# Type 2

# [1x,1y,4x,4y,3x,3y,2x,2y] in Yang's note. 
# 8x8 matrix
uxy_case2 = torch.tensor([[-1,1,1,1,1,-1,-1,-1],[-1,-1,-1,1,1,1,1,-1],
                          [1,1,-1,1,-1,-1,1,-1],[1,-1,1,1,-1,1,-1,-1],
                          [0,-1,0,1,0,-1,0,1], [-1,0,1,0,-1,0,1,0],
                          [1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]]) 

#for doublet, swap the (Ex,Ey) -> (Ey,Ex) (in the note)
uxy_case2_2 = torch.tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
                            [-1,0,1,0,-1,0,1,0],[0,-1,0,1,0,-1,0,1],
                            [0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]])




# Type 3
#[8x,8y,7x,7y,6x,6y,5x,5y,4x,4y,3x,3y,2x,2y,1x,1y]
# 16x16 matrix
uxy_case3 = torch.tensor([[0,1, -1,0, 1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1], [1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1, 0,1, 1,0],
                          [0,1, 1,0, 1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1],[1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1, 0,1, -1,0],
                          [0,1,1,0,-1,0,0,1, 0,-1, -1,0, 1,0, 0,-1], [1,0, 0,1, 0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0],
                          [0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0, 1,0, 0,1],[1,0, 0,-1, 0,1, 1,0, -1,0, 0,1, 0,-1, -1,0],
                          [0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],[-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],
                          [1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],[0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],
                          [0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],[0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],
                          [0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0],[0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1]]) 

#for doublet, swap the (Ex,Ey) -> (Ey,Ex) (in the note)
uxy_case3_2 = torch.tensor([[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0], [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                          [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0], [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                          [0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],[0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
                          [-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],[0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],
                          [0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],[1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],
                          [0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],[0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],
                          [0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1], [0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0]])  
#for doublet, swap the (Ex,Ey) -> (Ey,Ex)






# Type 1,ref
# Expected target vector is (16,1), ({All type 3 irrep from reference coordinate})
# Type 1,ref needs only 8 of them. 
# For the doublet, I only used two out of four doublet in type 3.
# The exact way is in the note.

uxy_case1_ref_choose = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0]])
uxy_case1_ref_choose_2 = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0]])



# Type 2,ref
# Essentially same with Type 1 ref
uxy_case2_ref_choose = uxy_case1_ref_choose
uxy_case2_ref_choose_2 = uxy_case1_ref_choose_2


# Type 3,ref
# For singlet, I need to use all of them since usual irrep in type 3 has all singlets.
# For doublet, I only used two out of four doublet in type 3.
# The exact way is in the note.
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
                                    [0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0]]) 

# print("Sanity check, expected 8x8 torch matrix:", uxy_case1.shape)
# print("Sanity check, expected 8x8 torch matrix:", uxy_case2.shape)
# print("Sanity check, expected 16x16 torch matrix:", uxy_case3.shape)




c4 = np.asarray([[0,-1],[1,0]])             # (a,b) --> (-b,a) but "clockwise 90" considering python arrangement. (e.x. [row, col] in python ~ [y,x] in physics) --> reason of irrep ordering
reflect = np.asarray([[1,0],[0,-1]])        # reflection around line y_id == 0


########## util function
#sorting
def sort_two(x1,x2):
    return tuple([min(x1,x2),max(x1,x2)])



# 2D index --> 1D index | After finding neighbor using 'cutoff radius' in 2D, save the result in 1D
def set_posid(pxy, width): #1D index --> 2D index
    return int(pxy[0]%width+(pxy[1]%width)*width)


# 1D index --> 2D index | To apply cutoff radius
def set_posxy(pos, width):
    return np.asarray([int(pos%width), int(pos//width)]) # compute x,y. pos is flattened array. So This code makes the 1D array as 2D.

def set_shift(posxy, prxy, height, width): # translating the idx number considering boundary condition
    [x,y] = prxy - posxy
    if x > width//2:
        x = - (width-x) #BD condition
    if x < -width//2:
        x = x%width
    if y > height//2:
        y = - (height-y) #BD condition
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
                        out = np.append(out, 2*set_posid(posxy+shift ,width)) # x
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1) # y
                        shift = np.dot(c4,shift)
                elif (xi==yi):
                    for i in range(4): #total 8 points
                        out = np.append(out, 2*set_posid(posxy+shift ,width)) # x
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1) # y
                        shift = np.dot(c4,shift)
                else:
                    shift_r = np.dot(c4, np.dot(reflect,shift)) # pi/8 change
                    for i in range(4): #total 16 points
                        out = np.append(out, 2*set_posid(posxy+shift ,width)) # x 
                        ref_point.append(2*set_posid(posxy+shift ,width)) # x
                        out = np.append(out, 2*set_posid(posxy+shift ,width)+1) # y
                        ref_point.append(2*set_posid(posxy+shift ,width)+1) # y
                        out = np.append(out, 2*set_posid(posxy+shift_r ,width)) # x
                        ref_point.append(2*set_posid(posxy+shift_r ,width)) # x
                        out = np.append(out, 2*set_posid(posxy+shift_r ,width)+1) # y
                        ref_point.append(2*set_posid(posxy+shift_r ,width)+1) # y
                        
                        
                        shift = np.dot(c4, shift) 
                        shift_r = np.dot(c4, shift_r)
            else:
                break
    
    # So it will return {current position x, current position y, (type 1 idx x,y), (type 2 idx x,y), (type 3 idx x,y)}
    return torch.tensor(out.astype(np.int16)), torch.tensor(np.array(ref_point).astype(np.int16))

def uxy_irrep_reference_matrix(cut_rad, height, width):
    cut_int = int(np.ceil(cut_rad)) # smallest "integer" r, s.t. r > cut_rad
    irrep_matrix = torch.tensor([[1,1], [1,-1]]) # reason is in note
    irrep_matrix2 = torch.tensor([[0,0], [0,0]]) # reason is in note
    #reference = torch.ones((1,16)) 
    type_count = 1
    reference_to_matrix = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                                     [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]]) # 2x16 matrix; for initial irrep (reason is in note)
    reference_to_matrix2 = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]) # reason is in note
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
                    #reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                    type_count += 1
                    
                elif (xi==yi):
                    irrep_matrix = torch.block_diag(irrep_matrix, uxy_case2)
                    irrep_matrix2 = torch.block_diag(irrep_matrix2, uxy_case2_2)
                    reference_to_matrix = torch.block_diag(reference_to_matrix, uxy_case2_ref_choose)
                    reference_to_matrix2 = torch.block_diag(reference_to_matrix2, uxy_case2_ref_choose_2)
                    #reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                    type_count += 1
                    
                else:
                    irrep_matrix = torch.block_diag(irrep_matrix, uxy_case3)
                    irrep_matrix2 = torch.block_diag(irrep_matrix2, uxy_case3_2)
                    reference_to_matrix = torch.block_diag(reference_to_matrix, uxy_case3_ref_choose)
                    reference_to_matrix2 = torch.block_diag(reference_to_matrix2, uxy_case3_ref_choose_2)
                    #reference = torch.cat((reference, torch.ones(1,16)), dim=0)
                    type_count += 1
                    
            else:
                break
                

    return irrep_matrix, irrep_matrix2, type_count, reference_to_matrix, reference_to_matrix2

#print("irrep matrix shape", uxy_irrep_reference_matrix(np.sqrt(20)+0.01,1,1).shape)


'''
In training code, it calls this class and give input of 'util.parameters' s.t. provide all dictionaries written in __init__, provides indexes and irreps.

Using this, 'descriptor_uxy' makes irrep from given 'displacement input' with the shape I described earlier.


'''
class Descriptors:
    def __init__(self, util_parameters, device, L): #util_parameters = util_matrix.pt -> It has all these dictionary things, device = cpu or gpu(make sure they are in same divice)
        self.neigh_id = util_parameters['neigh_id']
        self.ref_id = util_parameters['ref_id']
        self.uxy_irrep = util_parameters['uxy_irrep']
        self.uxy_irrep_2 = util_parameters['uxy_irrep_2']
        self.type_count = util_parameters['type_count']
        self.uxy_reference_to_matrix = util_parameters['uxy_reference_to_matrix']
        self.uxy_reference_to_matrix_2 = util_parameters['uxy_reference_to_matrix_2']
        self.L = L
        self.device = device

    def set_to_device(self):
        self.neigh_id = self.neigh_id.to(self.device)
        self.ref_id   = self.ref_id.to(self.device)
        self.uxy_irrep = self.uxy_irrep.to(self.device)
        self.uxy_irrep_2 = self.uxy_irrep_2.to(self.device)
        #self.uxy_ref = self.uxy_ref.to(self.device)
        self.uxy_reference_to_matrix = self.uxy_reference_to_matrix.to(self.device)
        self.uxy_reference_to_matrix_2 = self.uxy_reference_to_matrix_2.to(self.device)
        
        
    
    def sep_uxy(self, u): 
        
        uxy = u[self.neigh_id]
        uxy = torch.cat((uxy[:,:2]**2, uxy[:,2:]), dim=1) # I intentonally squared this, and it's in the note.
    
        return uxy

    ############ calculate descriptor
    def descriptor_uxy(self, u, uxy_config):
        uxy_case3 = torch.tensor([[0,1, -1,0, 1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1], [1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1, 0,1, 1,0],
                                  [0,1, 1,0, 1,0, 0,-1, 0,-1, -1,0, -1,0, 0,1],[1,0, 0,1, 0,-1, 1,0, -1,0, 0,-1, 0,1, -1,0],
                                  [0,1,1,0,-1,0,0,1, 0,-1, -1,0, 1,0, 0,-1], [1,0, 0,1, 0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0],
                                  [0,1, -1,0, -1,0, 0,-1, 0,-1, 1,0, 1,0, 0,1],[1,0, 0,-1, 0,1, 1,0, -1,0, 0,1, 0,-1, -1,0],
                                  [0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1, 0,1, 0,0],[-1,0, 0,0, 0,0, 1,0, -1,0, 0,0, 0,0, 1,0],
                                  [1,0, 0,0, 0,0, 1,0, 1,0, 0,0, 0,0, 1,0],[0,0, 0,1, 0,1, 0,0, 0,0, 0,1, 0,1, 0,0],
                                  [0,1, 0,0, 0,0, 0,-1, 0,1, 0,0, 0,0, 0,-1],[0,0, 1,0, -1,0, 0,0, 0,0, 1,0, -1,0, 0,0],
                                  [0,0,1,0,1,0,0,0, 0,0, 1,0, 1,0, 0,0],    [0,1,0,0,0,0,0,1, 0,1,0,0,0,0,0,1]], dtype=torch.float32).to(self.device)
        #Same tensor for type 3 irrep. I used this to make reference irrep from "averaged configuration data"
        
        # shape : matmul( (N * dim(F.V.) * (N,dim(F.V.)).t) ).t (in note)
        irrep = torch.matmul(self.uxy_irrep, uxy_config.t()).t()
        irrep_2 = torch.matmul(self.uxy_irrep_2, uxy_config.t()).t()
        
        uxy_value_ref = u[self.ref_id]
        uxy_ref_avg = uxy_value_ref.view(self.L*self.L,-1,16).sum(dim=1)

        irrep_ref = torch.matmul(uxy_case3, uxy_ref_avg.t()).t()
        irrep_ref_usage = irrep_ref.repeat(1, self.type_count)
        ref_irrep = torch.matmul(self.uxy_reference_to_matrix, irrep_ref_usage.t()).t()
        ref_irrep_2 = torch.matmul(self.uxy_reference_to_matrix_2, irrep_ref_usage.t()).t()
  
        irrep_mul_ref_irrep_1 = irrep * ref_irrep
        irrep_mul_ref_irrep_2 = irrep_2 * ref_irrep_2
       
       
        return irrep_mul_ref_irrep_1+irrep_mul_ref_irrep_2
    


    def cal_descriptor(self, u):
        uxy= self.sep_uxy(u)
        des_uxy = self.descriptor_uxy(u, uxy)
        return des_uxy


if __name__ == "__main__": #preventing re-running when it's called from other file(__name__ = "util_psq" if it's called from outside)
    neigh_list_shape = find_neighbor_id(0, cut, L, L)[0].shape
    ref_idx_shape = find_neighbor_id(0, cut, L, L)[1].shape

    neigh_id = torch.zeros(L*L,neigh_list_shape[0])
    ref_id = torch.zeros(L*L, ref_idx_shape[0])
    for i in range(L*L):
        neigh_id[i,:] = find_neighbor_id(i, cut, L, L)[0]
        ref_id[i,:] = find_neighbor_id(i, cut, L, L)[1]

    
    uxy_irrep, uxy_irrep_2, type_count, uxy_reference_to_matrix, uxy_reference_to_matrix_2 = uxy_irrep_reference_matrix(cut, L, L)
    #print(type_count)
    uxy_irrep = uxy_irrep.to(dtype=torch.float32)
    uxy_irrep_2 = uxy_irrep_2.to(dtype=torch.float32)
    #uxy_reference = uxy_reference.to(dtype=torch.float64)
    #type_count = type_count.to(dtype=torch.float64)
    uxy_reference_to_matrix = uxy_reference_to_matrix.to(dtype=torch.float32)
    uxy_reference_to_matrix_2 = uxy_reference_to_matrix_2.to(dtype=torch.float32)
    
    base_save = '/Users/jangho/Downloads/ML_peierls/util/'
    
    torch.save({"neigh_id": neigh_id.long(),"ref_id": ref_id.long(),"uxy_irrep": uxy_irrep.float(), "uxy_irrep_2":uxy_irrep_2.float(), "type_count":type_count, "uxy_reference_to_matrix":uxy_reference_to_matrix.float(), "uxy_reference_to_matrix_2":uxy_reference_to_matrix_2.float()},base_save+'util_matrix_66.pt')
