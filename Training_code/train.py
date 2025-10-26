import numpy as np
import torch
import torch.nn.functional as f
import torch.utils.data as data

from model.model import FCNN
from util.util_psq_batch import Descriptors_batch
import time

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight) #initialize the weights of torch.nn.Linear layers

from_bench_mark = False
torch.manual_seed(0) #seed for random number generation is deterministic
torch.set_default_dtype(torch.float64) #more accuracy

#gamma = 0.99
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Current device is ", device)
util_parameters = torch.load("/Users/jangho/Downloads/ML_Peierls/util/util_matrix_batch.pt")
Lsize=50
batch_size = 10
util_func = Descriptors_batch(util_parameters, device, Lsize,batch_size)
del util_parameters
util_func.set_to_device()

#print(util_func)

net = FCNN()
#net = torch.nn.Linear(10,1)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1-5)
loss_func = torch.nn.MSELoss()




data_path = "/Users/jangho/Downloads/ML_Peierls/data/"
u_tensor = torch.load(data_path + "your input file.pt", weights_only = True).double()
print("Total input data shape",u_tensor.shape)
#print("Total input data shape",spin_tensor.shape)
force_tensor = torch.load(data_path+"your output.pt", weights_only = True).double()
#temp_coordinate = torch.ones(L*L, dtype=torch.float64)
#x_temp = util_func.cal_descriptor(temp_coordinate)
#print("hey",x_temp.shape)
#print(net(x_temp))
# print("Time to operate one whole F.V.", end-start)


torch_data_set = data.TensorDataset(u_tensor, force_tensor)
loader = data.DataLoader(dataset=torch_data_set, batch_size=batch_size, shuffle=True)

torch.autograd.set_detect_anomaly(True)
time_start = time.time()
for epoch in range(1,100):
    fold = np.random.randint(0,10)
    for step, (u, force) in enumerate(loader):
        #if True:
        #print(u.shape)
        #print(force.shape)
        temp_coordinate = u[0].to(device).requires_grad_(True)
        #print(temp_coordinate.shape)
        optimizer.zero_grad()
        x_temp = util_func.cal_descriptor(temp_coordinate, batch_size)
        '''
        This code calculate 1. sum of local energy from descriptor, 2. calculate derivative using autograd to the sum 3. get the local force
        You can change this later part (e.x. output is (n, m, D,...) w/o summing whole lattice result) 
        '''
        energy_prediction_temp = net(x_temp)
        #print(energy_prediction_temp.shape)
        energy_prediction = torch.sum(energy_prediction_temp)
        force_prediction = -torch.autograd.grad(energy_prediction, temp_coordinate, create_graph=True)[0]
        
        #loss = gamma*loss_func(torch.cross(temp_coordinate,force_prediction),torch.cross(temp_coordinate,force[0])) + (1-gamma) *loss_func(force_prediction,force[0])
        #print(force[0])
        loss = loss_func(force_prediction, force[0])
        if step%100==fold:
        #if True:
            print("Answer : ",force[0])
            print("Predicted force : ",force_prediction)
            print("epoch: " + str(epoch) + "| Step: " + str(step) + '| Loss: ' + str(loss.cpu().detach().numpy()) + '| time: ' + str(time.time()-time_start))
        loss.backward()
        optimizer.step()
    if epoch%100==0 and epoch!=0:
        torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict()},"your model path"+str(epoch)+".pt")
        force_prediction_np = force_prediction.detach().cpu().numpy()
        force_np = force[0].detach().cpu().numpy()
        
        
        #np.savetxt(f"/Users/jangho/Downloads/ML_Peierls/save/M_(0,100,100)_3e-7_shuffle_epoch{epoch}.txt", force_prediction_np, fmt="%f")
        #np.savetxt(f"/Users/jangho/Downloads/ML_Peierls/save/A_(0,100,100)_3e-7_shuffle_epoch{epoch}.txt", force_np, fmt="%f")

