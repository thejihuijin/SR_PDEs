import torch
import torchvision
import torch.nn as nn
import math
import numpy as np
from SRNet import HFNet, HFDataset, ToTensor, train_net, display_state, normTensor

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import h5py
from torchvision import transforms

GPU = True
# Instantiate Net
net = HFNet(num_res_blocks=5)
if GPU:
    net = net.cuda()

# Load Training Data
print('Loading Training Data')
sr_data = h5py.File('SR_Datav2.h5','r')
transform = ToTensor()#transforms.Compose([ToTensor(),Normalize(means=np.zeros(4),stds=[1,1,1e-3,1e-3])])
#transform = normTensor()
traindata = HFDataset(sr_data,transform)
trainloader = DataLoader(traindata,batch_size=50,shuffle=True)

# Train Net
print('Beginning Training')
train_net(net,trainloader,num_epochs=50,GPU=GPU,save_epoch=5)


# Evaluate Net
print('Evaluating')
qtestdata = h5py.File('SR_Test_Quad_Datav2.h5','r')
#qtestinputs = qtestdata['training_data']
#qtestlabels = qtestdata['training_labels']
testdata = HFDataset(qtestdata,transform)
testloader = DataLoader(testdata,batch_size=100,shuffle=False)

n = len(testloader)
mses = np.zeros((n,4))
bil_mses = np.zeros((n,4))

for i, sample in enumerate(testloader):
    data, label = sample
    #label = label[0,...].numpy()
    label = label.numpy()
    # evaluate
    if GPU:
        output = net(data.cuda()).cpu().detach().numpy()
    else:
        output = net(data).detach().numpy()

    mses[i] = np.mean(np.power(output-label,2),axis=(0,-1,-2))

    # Compare against bil
    bil_mses[i] = np.mean(np.power(label,2),axis=(0,-1,-2))


print('Channel\tNet Error\tBilinear\tZOH')
for i,c in enumerate(['U','V','Px','Py']):
    print(c + '\t%.3fe-06\t%.3fe-06' % (mses[:,i].mean()*1e6, bil_mses[:,i].mean()*1e6))


    


