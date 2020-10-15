# -*- coding: utf-8 -*-
"""
@author: Youye
"""

import numpy as np
import time
import torch

import torch.optim as optim

from model_multiObject import Net

from ops_torch_gpu import  Numpy2Torch, InlierPortion, InlierLoss
from MultiObject_GenerateData import MultiObjectDataGenerator

# use GPU if it is available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%  Hyper-parameters

# random seed
np.random.seed(0)

# network parameters
inter_channel = 128;
numBlocks = 20

# loss function parameters
alpha1 = 1
alpha2 = 1;#2


# dataset parameters
seed = 1

numData = 200
max_inlierPortion = 0.3

# train      -  outlier portion random in range [0 - max_outlierPortion]
# validate   -  outlier portion [0 : 1/validate_size : max_outlierPortion]
# test       -  outlier portion  test_outlierPortion

train_size = 32
validate_size = 320
test_size = 320

# standard deviation of Gaussian noise unit pixels
pixel_std = 5

# training parameters
train_batch_size = 32
num_iter = int(train_size/train_batch_size)

learning_rate = 10**(-4);
num_epoch = 200


    
#%% Generate the dataset

for face_idx in range(20):
    
    inlier_face = face_idx #[0,1,...,19]
    inlier_angle = [0,180]

    DataGenerator =  MultiObjectDataGenerator(inlier_face, inlier_angle)
#
### train set
    [train_batch_Ps, train_batch_R,train_batch_T,train_batch_inlier] = DataGenerator.BatchGetData(train_size, numData,  pixel_std, max_inlierPortion ,seed = 1, mode='train' )
    TrainData = Numpy2Torch([train_batch_Ps,  train_batch_R , train_batch_T ,train_batch_inlier])
#np.savez('Train', train_batch_Ps=train_batch_Ps,  train_batch_weight= train_batch_weight,train_batch_R=train_batch_R,train_batch_T=train_batch_T,train_batch_inlier=train_batch_inlier)
#
### validate set
    [validate_batch_Ps,   validate_batch_R , validate_batch_T ,validate_batch_inlier] = DataGenerator.BatchGetData(validate_size, numData,  pixel_std, max_inlierPortion ,mode='validate' )
    ValidateData = Numpy2Torch([validate_batch_Ps, validate_batch_R , validate_batch_T , validate_batch_inlier])
#np.savez('Validate', validate_batch_Ps=validate_batch_Ps,  validate_batch_weight= validate_batch_weight,validate_batch_R=validate_batch_R,validate_batch_T=validate_batch_T,validate_batch_inlier=validate_batch_inlier)

#%% Define the network
        
    net = Net()
    net.cuda()


#%% create your optimizer

# define the optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)

    loss_record = [];

    print('Training start')
    
    for ep_iter in range(num_epoch):
        
        # shuffle the training data for each epoch
       
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        
        shuffled_Ps = TrainData[0][idx]
        shuffled_R = TrainData[1][idx]
        shuffled_T = TrainData[2][idx]
        shuffled_inlier = TrainData[3][idx]
        
        epoch_start_time = time.time()
        net.train()
        #train_loss = 0
        
        for i in range(num_iter):
               
            train_Ps = shuffled_Ps[i*train_batch_size:i*train_batch_size+train_batch_size,:,:,:].cuda() 
            train_inlier = shuffled_inlier[i*train_batch_size:i*train_batch_size+train_batch_size,:].cuda()
    
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
        
            train_predict_inlier = net(train_Ps)
            loss = InlierLoss(train_predict_inlier,train_inlier,alpha1 = alpha1,alpha2=alpha2)
            #loss = AngleLoss(train_predict_angle,train_xangle)
            loss.backward()
            optimizer.step()    # Does the update
            #train_loss += loss.data.item()
        
        print('The %d-th epoch takes %.1f seconds'%(ep_iter+1,time.time()-epoch_start_time) )
        
        
        # evaluate the trained model
        with torch.no_grad():
            net.eval()
        
            ## on validation set
            validate_Ps = ValidateData[0].cuda()
            validate_inlier = ValidateData[3].cuda()
            
        # predict the weight
            validate_predict_inlier = net(validate_Ps)
        
        # validation loss
            validate_loss = InlierLoss(validate_predict_inlier,validate_inlier,alpha1 = alpha1,alpha2=alpha2)
            
        # update the learning rate based on the network performance on the validation set 
            scheduler.step(validate_loss.detach().data.item()/validate_size)
    #        
    
            threshold = 0.7
        # result for overall
            por_inlier1 , gt_inlier_por1 = InlierPortion(validate_predict_inlier,validate_inlier,threshold)
            print('(Inlier): %.3f of gt inliers are detected, %.3f of the detected pair are inliers '%(por_inlier1.data.item(),gt_inlier_por1.data.item()))
            
    #    #     
            por_inlier11 , gt_inlier_por11 = InlierPortion(torch.ones([320,200]).cuda(),validate_inlier,threshold)
            print('(All one): %.3f of gt inliers are detected, %.3f of the detected pair are inliers '%(por_inlier11.data.item(),gt_inlier_por11.data.item()))
        

    
    torch.save(net.state_dict(),'200Epoch/'+str(face_idx)+'.pth')