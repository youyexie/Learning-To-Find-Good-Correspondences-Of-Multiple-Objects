# -*- coding: utf-8 -*-
"""
@author: Youye
"""
#%% load the input

import numpy as np

fileName = 'Data_3object0.3std1.npy'

x = np.load('TestData\\'+fileName)
x = np.float32(x)

#%% load the network

from FaceClassifier.model_multiObject import Net
import torch

device = torch.device("cuda")

Nets = {}
for i in range(20):
    Nets[i] = Net()
    Nets[i].load_state_dict(torch.load('FaceClassifier\\200Epoch\\'+str(i)+'.pth'))
    Nets[i].to(device)
    


#%% inference
    
elapsed_t = 0;

x = torch.tensor(x).cuda()

result_cat = Nets[0](x).detach().cpu().numpy()
result_cat = np.reshape(result_cat,[result_cat.shape[0],1,result_cat.shape[1]])

for i in range(1,20):
    with torch.no_grad():
        Nets[i].eval()
        
        temp = Nets[i](x).detach().cpu().numpy()
                
        temp = np.reshape(temp,[temp.shape[0],1,temp.shape[1]])
        result_cat = np.concatenate( (result_cat,temp) , axis=1)

#%% pose-processing the result
# every 3D-2D pair can only be assigned to one class

result = np.zeros(result_cat.shape)

for k in range(result.shape[0]):
    max_index = result_cat[k,:,:].argmax(axis=0)
    max_score = np.amax(result_cat[k,:,:],axis=0)

    for i in range(result_cat.shape[2]):
         result[k,max_index[i],i] = result_cat[k,max_index[i],i]        

# save the result
np.save('InferenceResult\\Inf_Our_'+fileName,result)


