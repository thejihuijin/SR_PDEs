import torch
import torchvision

import torch.nn as nn
from torch.nn.functional import upsample
import math

import numpy as np



#######################
# Model
#######################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)

        return out

def convBlock(inplanes,outplanes,k_size=3):
    padding = int((k_size-1)/2)
    return nn.Sequential(nn.Conv2d(inplanes,outplanes, kernel_size=k_size,stride=1,padding=padding,bias=False),
                                   nn.BatchNorm2d(outplanes),
                                   nn.ReLU(inplace=True))
class SRNet(nn.Module):
    def __init__(self,num_res_blocks=3,inplanes=4):
        super(SRNet,self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inplanes,64,kernel_size=5,stride=1,padding=2,bias=False),
                                    nn.PReLU())
        lr_layers = []
        for i in range(num_res_blocks):
            lr_layers.append(BasicBlock(64,64))
        
        self.lr_res_blocks = nn.Sequential(*lr_layers)
        
        # upsample
        self.upconv = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3,padding=1,output_padding=1,stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.PReLU())
        
        #hr_layers = []
        #for i in range(num_res_blocks):
        #    hr_layers.append(BasicBlock(128,128))
        
        self.hr_res_blocks = nn.Sequential(nn.Conv2d(64,64,kernel_size=1,stride=1,bias=False),
                                            nn.PReLU())
        self.conv_out = nn.Sequential(nn.Conv2d(64,inplanes,kernel_size=5,stride=1,padding=2,bias=False),
                                      nn.Tanh())
        
        
    def forward(self,x):
        residual = upsample(x,scale_factor=2,mode='bilinear')
        x = self.conv_in(x)
        x = self.lr_res_blocks(x)
        x = self.upconv(x)
        x = self.hr_res_blocks(x)
        x = self.conv_out(x)
        
        return x + residual

class HFNet(nn.Module):
    def __init__(self,num_res_blocks=3,inplanes=4,outplanes=3):
        super(HFNet,self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(inplanes,64,kernel_size=5,stride=1,padding=2,bias=False),
                                    nn.PReLU())
        lr_layers = []
        for i in range(num_res_blocks):
            lr_layers.append(BasicBlock(64,64))
        
        self.lr_res_blocks = nn.Sequential(*lr_layers)
        
        # upsample
        #self.upconv = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3,padding=1,output_padding=1,stride=2),
        #                            nn.BatchNorm2d(64),
        #                            nn.PReLU())
        
        #hr_layers = []
        #for i in range(num_res_blocks):
        #    hr_layers.append(BasicBlock(128,128))
        
        self.hr_res_blocks = nn.Sequential(nn.Conv2d(64,64,kernel_size=1,stride=1,bias=False),
                                            nn.PReLU())
        self.conv_out = nn.Sequential(nn.Conv2d(64,outplanes,kernel_size=5,stride=1,padding=2,bias=False),
                                      nn.Tanh())
        
        
    def forward(self,x):
        x = self.conv_in(x)
        x = self.lr_res_blocks(x)
        #x = self.upconv(x)
        x = self.hr_res_blocks(x)
        x = self.conv_out(x)
        
        return x 

class ASPPNet(nn.Module):
    def __init__(self,num_aspp_blocks=4,inplanes=4,outplanes=3,version=1):
        super(ASPPNet,self).__init__()
        self.num_aspp_blocks = num_aspp_blocks
        # input convolution
        self.conv_in = nn.Sequential(nn.Conv2d(inplanes,64,kernel_size=5,stride=1,padding=2,bias=False),
                                    nn.PReLU())
        
        # Generate features
        num_res_blocks=3
        lr_layers = []
        for i in range(num_res_blocks):
            lr_layers.append(BasicBlock(64,64))
        
        self.lr_res_blocks = nn.Sequential(*lr_layers)
        
        aspp_layers = []
        # support old version
        if version == 0:
            for i in range(num_aspp_blocks):
                aspp_layers.append(nn.Sequential(nn.Conv2d(64,16,kernel_size=3,padding=i+1,dilation=i+1,bias=False),
                                                nn.Conv2d(16,16,kernel_size=1,padding=0,dilation=1,stride=1),
                                                nn.Conv2d(16,16,kernel_size=1,padding=0,dilation=1,stride=1),
                                            nn.PReLU()))
        else:
            for i in range(num_aspp_blocks):
                aspp_layers.append(nn.Sequential(nn.Conv2d(64,32,kernel_size=3,padding=i+1,dilation=i+1,bias=False),
                                                nn.Conv2d(32,32,kernel_size=1,padding=0,dilation=1,stride=1),
                                            nn.PReLU()))
        # add aspp_layers to module children
        self.aspp_layers = aspp_layers
        for i, branch in enumerate(self.aspp_layers):
            self.add_module(str(i), branch)
        
        
        
        outlayers = num_aspp_blocks*16 if version == 0 else num_aspp_blocks*32
        self.hr_res_blocks = nn.Sequential(nn.Conv2d(outlayers,64,kernel_size=1,stride=1,bias=False),
                                            nn.PReLU())
        
        self.conv_out = nn.Sequential(nn.Conv2d(64,outplanes,kernel_size=5,stride=1,padding=2,bias=False),
                                      nn.Tanh())
        
        
    def forward(self,x):
        x = self.conv_in(x)
        x = self.lr_res_blocks(x)
        
        # ASPP
        x = torch.cat([b(x) for b in self.aspp_layers], 1)
        
        #x = self.upconv(x)
        x = self.hr_res_blocks(x)
        x = self.conv_out(x)
        
        return x 

class PSNet(nn.Module):
    def __init__(self,num_res_blocks=3, inplanes=4, upsample=2, res=True):
        super(PSNet,self).__init__()
        self.res = res
        self.conv_in = nn.Sequential(nn.Conv2d(inplanes,64,kernel_size=5,stride=1,padding=2,bias=False),
                                    nn.PReLU())
        lr_layers = []
        for i in range(num_res_blocks):
            lr_layers.append(BasicBlock(64,64))
        
        self.lr_res_blocks = nn.Sequential(*lr_layers)
        
        # upsample
        #self.upconv = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3,padding=1,output_padding=1,stride=2),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(inplace=True))
        self.upconv = nn.Sequential(nn.Conv2d(64,inplanes*upsample*upsample,kernel_size=3,stride=1,padding=1,bias=False),
                                    nn.PixelShuffle(upsample),
                                    nn.PReLU())
        #hr_layers = []
        #for i in range(num_res_blocks):
        #    hr_layers.append(BasicBlock(128,128))
        
        self.hr_res_blocks = nn.Conv2d(inplanes,inplanes,kernel_size=1,stride=1,bias=False)
        #self.conv_out = nn.Sequential(nn.Conv2d(64,4,kernel_size=5,stride=1,padding=2,bias=False),
        #nn.Tanh())
        
        
    def forward(self,x):
        if self.res:
            residual = upsample(x,scale_factor=2,mode='bilinear')
        x = self.conv_in(x)
        x = self.lr_res_blocks(x)
        x = self.upconv(x)
        x = self.hr_res_blocks(x)
        #x = self.conv_out(x)
        if self.res:
            return x + residual
        return x





#######################
# Dataloader
#######################
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

class HFDataset(Dataset):
    def __init__(self,h5py_data, transform=None):
        self.training_data = h5py_data['training_data']
        self.training_labels = h5py_data['training_labels']
        self.transform = transform
    def __len__(self):
        return self.training_data.shape[0]
    
    def __getitem__(self,idx):
        sample = (self.training_data[idx,...].astype(np.float),
                  self.training_labels[idx,...].astype(np.float))
        if self.transform:
            sample = self.transform(sample)
        inputdata, label = sample
        inputdata = nn.functional.upsample(inputdata.unsqueeze(0),scale_factor=2,mode='bilinear')[0,...]
        label = label - inputdata

        return (inputdata,label)

class NSDataset(Dataset):
    def __init__(self,h5py_data, transform=None):
        self.training_data = h5py_data['training_data']
        self.training_labels = h5py_data['training_labels']
        self.transform = transform
    def __len__(self):
        return self.training_data.shape[0]
    
    def __getitem__(self,idx):
        sample = (self.training_data[idx,...].astype(np.float),
                  self.training_labels[idx,...].astype(np.float))
        if self.transform:
            return self.transform(sample)
        return sample   
class ToTensor(object):
    def __call__(self,sample):
        data,label = sample
        return (torch.from_numpy(data).float(),
                torch.from_numpy(label).float())
class Normalize(object):
    def __init__(self,means,stds):
        self.normalizer = transforms.Normalize(means,stds)
    def __call__(self,sample):
        data,label = sample
        data = self.normalizer(data)
        label = self.normalizer(label)
        return (data,label)
def normTensor():
    return transforms.Compose([ToTensor(),Normalize(means=np.zeros(4),stds=[1,1,1e-3,1e-3])])

def display_state(state):
    plt.figure(figsize=(10,10))
    titles = ['U','V','P_x','P_y']
    state = state.detach().numpy()
    
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(state[i,...])
        plt.colorbar()
        plt.title(titles[i])

#####################
# Training
#####################
import torch.optim as optim

import datetime
import os
import time

def get_datetime():
    time = datetime.datetime.today()
    out = str(time.date())+'_'+str(time.hour).zfill(2)
    out = out+'-'+str(time.minute).zfill(2)
    out = out+'-'+str(time.second).zfill(2)
    return out

def train_net(net,trainloader,GPU=False,num_epochs=10,lr=.0002,
              weightpath = './weights/', save_epoch=50,
              saveweights = True):
    # Create output directory
    weightpath = os.path.join(weightpath,get_datetime())
    os.makedirs(weightpath)
    logpath = os.path.join(weightpath,'log.txt')
    with open(logpath, "wt") as text_file:
        print('Epoch\tLoss\tEpoch Time\tTotal Time',file=text_file)

    num_data = len(trainloader)*trainloader.batch_size
    
    # Accumulate Log text
    logtxt = ''
    
    # Determine Minibatch size
    minibatch=max(1,int(len(trainloader)/10))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=lr,betas=(.5,.999),weight_decay=0)
    trainstart = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        running_count = 0
        epochstart = time.time()
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            if GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()


            # zero gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            running_count += 1
            if (i+1) % minibatch == 0:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.5f, %.2f seconds elapsed' %
                      (epoch + 1, i + 1, running_loss / running_count, time.time() - epochstart ))
                running_loss = 0.0
                running_count = 0
        epochend = time.time()
        print('Epoch %d Training Time: %.2f seconds\nTotal Elapsed Time: %.2f seconds' %
               (epoch+1, epochend-epochstart,epochend-trainstart))
        
        # write loss to logfile
        #with open(logpath, "at") as text_file:
        #    print('%i\t%f\t%f\t%f\n' % 
        #        (epoch+1,float(epoch_loss)/num_data,epochend-epochstart,epochend-trainstart)
        #         ,file=text_file)
        logtxt += '%i\t%f\t%f\t%f\n' % (epoch+1,float(epoch_loss)/len(trainloader),epochend-epochstart,epochend-trainstart)
        epoch_loss=0.0

        
        # Save weights
        if (epoch % save_epoch == 0 or epoch == num_epochs-1):
            if saveweights:
                outpath = os.path.join(weightpath,'epoch_'+str(epoch+1)+'.weights')
                net = net.cpu()
                torch.save(net.state_dict(),outpath)
                if GPU:
                    net = net.cuda()
            
            # write loss to logfile
            with open(logpath, "at") as text_file:
                print(logtxt[:-2],file=text_file)
                logtxt = ''

    print('Finished Training')
def convert_vector(batch):
    magnitudes = torch.sqrt(batch[:,0,...].pow(2)+batch[:,1,...].pow(2))
    angles = torch.atan2(batch[:,1,...],batch[:,0,...])/np.pi
    return torch.cat((magnitudes.unsqueeze(1),angles.unsqueeze(1)),dim=1)
def tv_loss(y):
    return torch.sum(torch.abs(y[...,:-1]-y[...,1:])) + torch.sum(torch.abs(y[...,:-1,:]-y[...,1:,:]))
def train_net_vecloss(net,trainloader,GPU=False,num_epochs=10,lr=.0002,
              weightpath = './weights/', save_epoch=50, tv=False, tv_c=2, tv_reg=.1,
              saveweights = True):
    # Create output directory
    weightpath = os.path.join(weightpath,get_datetime())
    os.makedirs(weightpath)
    logpath = os.path.join(weightpath,'log.txt')
    with open(logpath, "wt") as text_file:
        print('Epoch\tLoss\tEpoch Time\tTotal Time',file=text_file)

    num_data = len(trainloader)*trainloader.batch_size
    
    # Accumulate Log text
    logtxt = ''
    
    # Determine Minibatch size
    minibatch=max(1,int(len(trainloader)/10))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=lr,betas=(.5,.999),weight_decay=0)
    trainstart = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        running_count = 0
        epochstart = time.time()
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            if GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()


            # zero gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            
            # Convert to vector magnitude and angle
            labels = labels + inputs[:,:-1,...]
            outputs = outputs + inputs[:,:-1,...]
            label_vecs = convert_vector(labels)
            output_vecs = convert_vector(outputs)
            
            loss = .1*criterion(output_vecs,label_vecs) + criterion(outputs,labels)
            if tv:
                loss =loss + tv_reg*tv_loss(outputs[:,tv_c,...])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            running_count += 1
            if (i+1) % minibatch == 0:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.5f, %.2f seconds elapsed' %
                      (epoch + 1, i + 1, running_loss / running_count, time.time() - epochstart ))
                running_loss = 0.0
                running_count = 0
        epochend = time.time()
        print('Epoch %d Training Time: %.2f seconds\nTotal Elapsed Time: %.2f seconds' %
               (epoch+1, epochend-epochstart,epochend-trainstart))
        
        # write loss to logfile
        #with open(logpath, "at") as text_file:
        #    print('%i\t%f\t%f\t%f\n' % 
        #        (epoch+1,float(epoch_loss)/num_data,epochend-epochstart,epochend-trainstart)
        #         ,file=text_file)
        logtxt += '%i\t%f\t%f\t%f\n' % (epoch+1,float(epoch_loss)/len(trainloader),epochend-epochstart,epochend-trainstart)
        epoch_loss=0.0

        
        # Save weights
        if (epoch % save_epoch == 0 or epoch == num_epochs-1):
            if saveweights:
                outpath = os.path.join(weightpath,'epoch_'+str(epoch+1)+'.weights')
                net = net.cpu()
                torch.save(net.state_dict(),outpath)
                if GPU:
                    net = net.cuda()
            
            # write loss to logfile
            with open(logpath, "at") as text_file:
                print(logtxt[:-2],file=text_file)
                logtxt = ''

    print('Finished Training')


        
    
