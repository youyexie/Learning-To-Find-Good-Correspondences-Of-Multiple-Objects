# -*- coding: utf-8 -*-
"""
@author: Youye
"""


import torch
import torch.nn as nn

def weightedDLTmatrix(Ps,weight):
    """
    Input:  Ps = [1,1,N,5 (X,Y,Z,x,y) ], Weights = [1,N]
            P_world-3Dpoint (N,[X,Y,Z])  p_img-2Dpoint (N,[x,y])  Weight (N,)
    OutpuT: weighted DLT matrix A^T*W*A
    """
    # DLT matrix
    
    N = Ps.shape[2]
    A = torch.zeros([2*N,12])
    W = torch.eye(2*N)
    
    for i in range(N):
        
        A[2*i:(2*i+2),:] = OneCorrespondMatrix(Ps[0:1,0:1,i:i+1,:])    
                
        W[2*i:(2*i+1),2*i:(2*i+1)] = weight[0,i]
        W[(2*i+1):(2*i+2),(2*i+1):(2*i+2)] = weight[0,i]
         
    
    return torch.matmul( torch.t(A),torch.matmul(W,A) )


def DLTmatrix(Ps):
    """
    Input:  Ps = [1,1,N,5]
            P_world-3Dpoint (N,[X,Y,Z])  p_img-2Dpoint (N,[x,y])  Weight (N,)
    OutpuT: DLT matrix A
    """
    # DLT matrix
    
    N = Ps.shape[2]
    A = torch.zeros([2*N,12])
    
    for i in range(N):
        
        A[2*i:(2*i+2),:] = OneCorrespondMatrix(Ps[0:1,0:1,i:i+1,:])             
    
    return A
    
    
def OneCorrespondMatrix(P):
    """
    Input: 3D-2D Point pair (1,1,1,5)   5 - [X,Y,Z,x,y]
    OutpuT: DLT matrix for 1 pair of correspondence
    """
    [ X,Y,Z ] =  P[0,0,0,:][0:3]
    [ x,y ] =  P[0,0,0,:][3:5]
    return torch.tensor([[X,Y,Z,1,0,0,0,0,-x*X,-x*Y,-x*Z,-x],
                     [0,0,0,0,X,Y,Z,1,-y*X,-y*Y,-y*Z,-y]])
  

def EigenLoss(batch_Ps,batch_weight,batch_e,alpha1 = 1,alpha2=1,beta=0.005 ):
    """
    Input: Ps([Batch_size,1,N,[X,Y,Z,x,y]])  weight (Batch_size,N)  e (Batch_size,12)   
    OutpuT: Loss function in 'Eigendecomposition-free..' paper
    """    
    batch_size = batch_Ps.shape[0]
    Loss = torch.zeros(batch_size)
    term_1 = torch.zeros(batch_size)
    term_2 = torch.zeros(batch_size)
    
    for i in range(batch_size):
        
        # extract the parameters
        weight = batch_weight[i:i+1,:]
        e = batch_e[i:i+1,:]
        Ps = batch_Ps[i:i+1,0:1,:,:]
        
        # calculate the loss 
        ATWA = weightedDLTmatrix(Ps,weight)
        e_orth = torch.eye(12) - torch.matmul(torch.t(e),e)
        
        term_1[i] = alpha1*torch.matmul(torch.matmul(e,ATWA),torch.t(e)) 
        term_2[i] = alpha2*torch.exp(-beta*( torch.trace( torch.matmul(torch.matmul(e_orth,ATWA),e_orth)  ) ) )
         
        # cumulate the loss
        Loss[i] = term_1[i]+term_2[i]
        
        
    return torch.sum(Loss)


def EigenLoss_test(batch_Ps,batch_weight,batch_e,alpha1 = 1,alpha2 = 1 ,beta=0.005 ):
    """
    Input: Ps([Batch_size,1,N,[X,Y,Z,x,y]])  weight (Batch_size,N)  e (Batch_size,12)   
    OutpuT: Loss function in 'Eigendecomposition-free..' paper
    """    
    batch_size = batch_Ps.shape[0]
    Loss = torch.zeros(batch_size)
    term_1 = torch.zeros(batch_size)
    term_2 = torch.zeros(batch_size)
    
    for i in range(batch_size):
        
        # extract the parameters
        weight = batch_weight[i:i+1,:]
        e = batch_e[i:i+1,:]
        Ps = batch_Ps[i:i+1,0:1,:,:]
        
        # calculate the loss 
        ATWA = weightedDLTmatrix(Ps,weight)
        e_orth = torch.eye(12) - torch.matmul(torch.t(e),e)
        
        term_1[i] = alpha1*torch.matmul(torch.matmul(e,ATWA),torch.t(e)) 
        term_2[i] = alpha2*torch.exp(-beta*( torch.trace( torch.matmul(torch.matmul(e_orth,ATWA),e_orth)  ) ) )
         
        # cumulate the loss
        Loss[i] = term_1[i]+term_2[i]
        
        
    return [ ATWA , Loss, term_1,term_2]
    
def Numpy2Torch(ListIn):
    """
    Input: a list of numpy array
    Output: a list of torch tensor
    """
    ListOut = list()
    
    for i in range(len(ListIn)): 
        ListOut.append( torch.from_numpy(ListIn[i]).float() )
        
    return ListOut



def EigenLoss_SingleObject (batch_Ps,batch_weight,batch_R,batch_t,alpha1 = 1,alpha2 = 1 ,beta=0.005 ):
    """
    Input: Ps([Batch_size,1,N,[X,Y,Z,x,y]])  weight (Batch_size,N)  R (Batch_size,3,3) t(Batch_size,3,1)   
    OutpuT: Loss function in 'Eigendecomposition-free training of deep networks with zero eigenvalue-based losses' paper
    """        

    Xw = torch.unsqueeze(batch_Ps[:,0,:,0],dim=-1)  #[batch_size, N,1]
    Yw = torch.unsqueeze(batch_Ps[:,0,:,1],dim=-1)
    Zw = torch.unsqueeze(batch_Ps[:,0,:,2],dim=-1)
    
    ones = torch.ones_like(Xw)
    zeros = torch.zeros_like(Xw)
    
    u = torch.unsqueeze(batch_Ps[:,0,:,3],dim=-1)
    v = torch.unsqueeze(batch_Ps[:,0,:,4],dim=-1)
    
    M1n = torch.cat([Xw, Yw, Zw, ones, zeros, zeros, zeros, zeros, -u*Xw, -u*Yw, -u*Zw, -u],dim=2)  #[ batch_size, N , 12 ]
    M2n = torch.cat([zeros, zeros, zeros, zeros, Xw, Yw, Zw, ones, -v*Xw, -v*Yw, -v*Zw, -v],dim=2)    
    M = torch.cat([M1n, M2n], dim=1) # batch_size, 2*N, 12
    
    w_2n = torch.cat([batch_weight,batch_weight],dim=-1) #[ batch_size, 2*N ]
    M_t = M.permute([0,2,1])
    wM = torch.unsqueeze(w_2n,dim=-1)*M  #[ batch_size, 2*N , 12 ]
    MwM = torch.matmul(M_t,wM) #[ batch_size, 12 , 12 ]
    
    # normalized the e vector
    e_gt = torch.reshape(torch.cat([batch_R, batch_t], dim=-1), (-1,12)) # bs, 12
    e_gt /= torch.norm(e_gt,dim=1,keepdim=True)
    e_gt = torch.unsqueeze(e_gt,dim=-1) #[ batch_size, 12 , 1 ]
    
    e_gt_t = e_gt.permute([0,2,1])
    d_term = torch.matmul(torch.matmul(e_gt_t,MwM),e_gt)
    
    e_hat = torch.eye(12).reshape((1, 12, 12)).repeat(batch_Ps.shape[0], 1, 1).to(batch_Ps.device) - torch.matmul(e_gt,e_gt_t)
    e_hat_t = e_hat.permute([0,2,1])
    XwX_e_neg = torch.matmul(torch.matmul(e_hat_t,MwM),e_hat)
    r_term = torch.sum( torch.diagonal(XwX_e_neg,dim1=1,dim2=2),axis=1 )

    return alpha1*torch.sum(d_term)+alpha2*torch.sum( torch.exp(-beta*r_term) )    
    
def AngleLoss(predict_angle,gt_angle):
    [batch_size,_,numData] = predict_angle.shape
    return -torch.sum(predict_angle*gt_angle)/(batch_size*numData)

def InlierLoss(predict_inlier,gt_inlier,alpha1 = 1,alpha2=1):
    #loss = nn.BCELoss(weight=torch.abs((gt_inlier-1)*40-1),reduction='mean')#
    #CE_term = loss(predict_inlier, gt_inlier)
    inlier_mask = (gt_inlier==1)
    outlier_mask = (gt_inlier==0)
    
    num_inlier = torch.sum(inlier_mask)
    num_outlier = torch.sum(outlier_mask)
    
    inlier_part = -torch.sum( inlier_mask*torch.log(predict_inlier+torch.tensor(1e-12)) )/num_inlier
    outlier_part =  -torch.sum( outlier_mask*torch.log(1-predict_inlier+torch.tensor(1e-12)) )/num_outlier
    
    return alpha1*inlier_part+alpha2*outlier_part
    
    
def EigenLoss_MultiObject(batch_Ps,batch_weight,batch_R,batch_t,predict_inlier,gt_inlier,alpha1 = 1,alpha2 = 1 ,alpha3 = 1,beta=0.005):
    """
    Input: Ps([Batch_size,1,N,[X,Y,Z,x,y]])  weight (Batch_size,8,N)  R (Batch_size,8,3,3) t(Batch_size,8,3,1)   
    OutpuT: Extended version of the loss function in 'Eigendecomposition-free training of deep networks with zero eigenvalue-based losses' paper
    """        
    # the first [batch_size, 1, N] weight is the inlier indicator
    loss = nn.BCELoss(weight=torch.abs((gt_inlier-1)*50-1),reduction='mean')#
    CE_term = loss(predict_inlier, gt_inlier)
    
    
    [ batch_size , _ , N , _ ] = batch_Ps.shape
    num_obj = batch_weight.shape[1]
       
    Xw = torch.unsqueeze(batch_Ps[:,0,:,0],dim=-1)  #[batch_size, N,1]
    Yw = torch.unsqueeze(batch_Ps[:,0,:,1],dim=-1)  #[batch_size, N,1]
    Zw = torch.unsqueeze(batch_Ps[:,0,:,2],dim=-1)  #[batch_size, N,1]
    
    ones = torch.ones_like(Xw)    #[batch_size, N,1]
    zeros = torch.zeros_like(Xw)  #[batch_size, N,1]
    
    u = torch.unsqueeze(batch_Ps[:,0,:,3],dim=-1)  #[batch_size, N,1]
    v = torch.unsqueeze(batch_Ps[:,0,:,4],dim=-1)  #[batch_size, N,1]
    
    M1n = torch.cat([Xw, Yw, Zw, ones, zeros, zeros, zeros, zeros, -u*Xw, -u*Yw, -u*Zw, -u],dim=2)  #[ batch_size, N , 12 ]
    M2n = torch.cat([zeros, zeros, zeros, zeros, Xw, Yw, Zw, ones, -v*Xw, -v*Yw, -v*Zw, -v],dim=2)    
    M = torch.cat([M1n, M2n], dim=1).view(batch_size,1,2*N,12) # [ batch_size, 2*N, 12 ]  to [ batch_size, 1, 2*N, 12 ]
    M = M.repeat(1,num_obj,1,1)    # [ batch_size, 8, 2*N, 12 ]
    
    w_2n = torch.unsqueeze(torch.cat([batch_weight,batch_weight],dim=-1),dim=-1) #[ batch_size, 8 , 2*N,1 ]
    M_t = M.permute([0,1,3,2]) #[ batch_size, 8 , 12 , 2*N ]
    wM = w_2n*M  #[ batch_size, 8 , 2*N , 12 ]
    MwM = torch.matmul(M_t,wM) #[ batch_size, 8 , 12 , 12 ]
    
    # normalized the e vector
    e_gt = torch.reshape(torch.cat([batch_R, batch_t], dim=-1), (batch_size,num_obj,12))  #[ batch_size, 8 , 3 , 4 ]
    e_gt /= torch.norm(e_gt,dim=2,keepdim=True)
    e_gt = torch.unsqueeze(e_gt,dim=-1) #[ batch_size , 8 , 12 , 1 ]
    
    e_gt_t = e_gt.permute([0,1,3,2]) #[ batch_size , 8 , 1 , 12 ]
    d_term = torch.matmul(torch.matmul(e_gt_t,MwM),e_gt) #[ batch_size , 8 , 1 , 1 ]
    
    e_hat = torch.eye(12).reshape((1, 1, 12, 12)).repeat(batch_size, num_obj , 1, 1).to(batch_Ps.device) - torch.matmul(e_gt,e_gt_t) #[ batch_size, 8 , 12 , 12 ]
    e_hat_t = e_hat.permute([0,1,3,2]) #[ batch_size, 8 , 12 , 12 ]
    XwX_e_neg = torch.matmul(torch.matmul(e_hat_t,MwM),e_hat) #[ batch_size, 8 , 12 , 12 ]
    r_term = torch.sum( torch.diagonal(XwX_e_neg,dim1=2,dim2=3),axis=2 )

    return alpha1*CE_term + alpha2*torch.mean(d_term)+alpha3*torch.mean( torch.exp(-beta*r_term) )    
    

def InlierPortion(predict_weight,gt_weight,threshold=0.3):
    """
    Input: predict_weight ([Batch_size,N]), gt_weight ([Batch_size,N]) , threshold for deciding inliers
    OutpuT: portion of the ground truth inlier detected, ground truth inliers among the dtected inliers
    """    
    predict_weight[predict_weight >= threshold] = 1
    predict_weight[predict_weight < threshold] = 0
    
    por_inlier = torch.sum(predict_weight*gt_weight)/torch.sum(gt_weight)
    gt_inlier_por = torch.sum(predict_weight*gt_weight)/torch.sum(predict_weight)
    
    return por_inlier , gt_inlier_por
    
    


