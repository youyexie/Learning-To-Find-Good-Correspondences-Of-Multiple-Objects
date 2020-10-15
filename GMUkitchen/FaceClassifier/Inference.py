# -*- coding: utf-8 -*-
"""
@author: Youye
"""

#%% load the input

import scipy.io
import numpy as np

fileName = 'match76'

mat = scipy.io.loadmat('InputData\\'+fileName+'.mat')
x = mat['input'];
x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
x = np.float32(x)


#%% load the network

from model_multiObject import Net
import torch

device = torch.device("cuda")

Nets = {}
for i in range(20):
    Nets[i] = Net()
    Nets[i].load_state_dict(torch.load('200Epoch/'+str(i)+'.pth'))
    Nets[i].to(device)
    Nets[i].eval()


#%% inference

x = torch.tensor(x).cuda()

result_cat = Nets[0](x).detach().cpu().numpy()

for i in range(1,20):
    
    temp = Nets[i](x).detach().cpu().numpy()
    result_cat = np.concatenate( (result_cat,temp) , axis=0)

#%% pose-processing the result
# every 3D-2D pair can only be assigned to one class

result = np.zeros(result_cat.shape)

threshold = 0.9

max_index = result_cat.argmax(axis=0)
max_score = np.amax(result_cat,axis=0)

for i in range(result_cat.shape[1]):
     result[max_index[i],i] = result_cat[max_index[i],i]
        
    
np.save('InferenceResult\\'+fileName+'.npy',result)


