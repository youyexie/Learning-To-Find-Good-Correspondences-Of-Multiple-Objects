# -*- coding: utf-8 -*-
"""
@author: Youye
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#%% Define the ResNet block

class ResNetBlock(nn.Module):
    
    def __init__(self,inter_channel = 128):
        
        super(ResNetBlock, self).__init__()
        
        
        # RESNET Block Operator        
        self.conv1 = nn.Conv2d( in_channels=inter_channel , out_channels=inter_channel, kernel_size=(1,1))
        self.CN1 = nn.InstanceNorm2d( num_features=inter_channel  )
        self.BN1 = nn.BatchNorm2d( num_features=inter_channel , affine=False )
        
        self.conv2 = nn.Conv2d( in_channels=inter_channel , out_channels=inter_channel, kernel_size=(1,1))
        self.CN2 = nn.InstanceNorm2d( num_features=inter_channel  )
        self.BN2 = nn.BatchNorm2d( num_features=inter_channel , affine=False )
         
    def forward(self,x):
        
        # define the structure of the ResNetBlock
        identity = x;
        x = self.conv1(x);  #print(x.size())
        x = self.CN1(x);    #print(x.size())
        x = self.BN1(x);    #print(x.size())
        x = F.relu(x);      # print(x.size())
        
        x = self.conv2(x);
        x = self.CN2(x);
        x = self.BN2(x);
        x = F.relu(x);        
        
        x = x + identity;
        
        return x

#% Define the network structure
        
class Net(nn.Module):

    def __init__(self, numBlocks1 = 5, numBlocks2=19,inter_channel=128):
        self.numBlocks1 = numBlocks1  # for inlier predictor
        self.numBlocks2 = numBlocks2  # for object weight predictor
        
        super(Net, self).__init__()
        
        # INPUT layer operator
        self.convInt = nn.Conv2d( in_channels=1 , out_channels=inter_channel , kernel_size=(1,5) )
        
        # Common ResNetBlock 
        layers1 = []        
        
        for _ in range(0,self.numBlocks1):
            layers1.append( ResNetBlock(inter_channel) )
            
        self.ResNet1 = nn.Sequential(*layers1)    
  

        # OUTPUT layer operator 
        self.convInlierPredictor = nn.Conv2d( in_channels=inter_channel , out_channels=1, kernel_size=(1,1) )
        
    def forward(self, x):
        # Input Layer
        x = self.convInt(x)
        
        # ResNet blocks
        x = self.ResNet1(x) 
        
        
######### inlier predictor ################

        
        # [ Batch_size ,  128 , num_weight, 1 ]     
        [batch_size, _, numData,_] = x.shape
      
        # inlier predictor
        x = self.convInlierPredictor(x)
        x = x.view([batch_size,numData])
        
        
        x = torch.tanh(x)
        x = F.relu(x)        
        
        return x
    
    
#% Define the network structure
        
class AngleNet(nn.Module):

    def __init__(self, numBlocks = 20,inter_channel=128):
        self.numBlocks = numBlocks
        super(AngleNet, self).__init__()
        
        # INPUT layer operator
        self.convInt = nn.Conv2d( in_channels=1 , out_channels=inter_channel , kernel_size=(1,5) )
        
        # Common ResNetBlock 
        layers = []        
        
        for _ in range(0,self.numBlocks):
            layers.append( ResNetBlock(inter_channel) )
            
        self.ResNet = nn.Sequential(*layers)          
         
           
        # OUTPUT layer operator 
        self.convOut = nn.Conv2d( in_channels=inter_channel , out_channels=9, kernel_size=(1,1) )
        
        self.SoftMax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input Layer
        x = self.convInt(x)
        
        # ResNet blocks
        x = self.ResNet(x) 

        
        # [ Batch_size ,  128 ,numData, 1 ]     
        [batch_size, _, numData,_] = x.shape
      
        # [ Batch_size ,  9 ,numData, 1 ]     
        x = self.convOut(x)
        x = x[:,:,:,0]  
        
        x = self.SoftMax(x)
        
        return x