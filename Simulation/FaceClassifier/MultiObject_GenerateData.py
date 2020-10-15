# -*- coding: utf-8 -*-
"""
@author: Youye
"""
import numpy as np
from functools import reduce

"""
    
For outlier:
    randomly different R and t
    
Note:
    keep everything in float32 to avoid precision issue when using pytorch
"""

NorVec = np.load('NorVec.npy')
FaceDict = np.load('FaceDict.npz',allow_pickle = True)
FaceDict = FaceDict['FaceDict'][()]


def findIndex(RangeList,angle):
    idx = 0
    for i in RangeList:
        if angle >= i:
            idx = idx + 1;
    return idx


def SampleAxis( bin_idx=0 , FaceDict = FaceDict):
    '''
    Randomly sampled a rotation axis within the specific facet bin
    '''
    weight = np.random.uniform(0,1,3)
    # resample if boundary is achieved
    while np.sum(weight==0)>=1:
        weight = np.random.uniform(0,1,3)
        
    # sum up to 1
    weight = weight/weight.sum()
    
    # use the weight to get the axis within specific face bin
    k = weight[0]*FaceDict[bin_idx][0] + weight[1]*FaceDict[bin_idx][1] + weight[2]*FaceDict[bin_idx][2]
    k = k/np.linalg.norm(k)
    
    return k


def DistMeasure(k,NorVec):
    '''
    find which bin that k belongs to
    '''
    k = np.reshape(k,[1,3])
    idx = np.argmin( np.linalg.norm(k - NorVec,axis=1) )
    return idx



def RotationMat(k,theta):
    '''
    k - normalized rotation axis
    theta - rotation angle (unit: degree)
    '''
    kx = k[0]; ky = k[1]; kz = k[2];
    c = np.cos(np.deg2rad(theta))
    s = np.sin(np.deg2rad(theta))
    v = 1 - np.cos(np.deg2rad(theta))

    R = np.zeros([3,3])
    R[0,0] = kx*kx*v+c;     R[0,1] = kx*ky*v-kz*s; R[0,2] = kx*kz*v+ky*s;
    R[1,0] = kx*ky*v+kz*s;  R[1,1] = ky*ky*v+c;    R[1,2] = ky*kz*v-kx*s;
    R[2,0] = kx*kz*v-ky*s;  R[2,1] = ky*kz*v+kx*s; R[2,2] = kz*kz*v+c;

    return R

def findAngleAxis(R):
    '''
    R - rotation matrix
    '''
    theta =  np.arccos( (R[0,0]+R[1,1]+R[2,2]-1)/2 ) 
    
    k = np.zeros(3)
    k[0] = 1/(2*np.sin(theta))*(R[2,1] - R[1,2])
    k[1] = 1/(2*np.sin(theta))*(R[0,2] - R[2,0])
    k[2] = 1/(2*np.sin(theta))*(R[1,0] - R[0,1])
    
    return k , np.rad2deg(theta)

def Rotation3DMatrix(Xangle=0,Yangle=0,Zangle=0):
    """
    unit: degree [0,360]
    """
    Xangle = np.pi*Xangle/180
    Yangle = np.pi*Yangle/180
    Zangle = np.pi*Zangle/180
        
    Rx = np.array([[1,0,0],
                   [0,np.cos(Xangle),-np.sin(Xangle)],
                   [0,np.sin(Xangle),np.cos(Xangle)]])
    Ry= np.array([[np.cos(Yangle),0,np.sin(Yangle)],
                   [0,1,0],
                   [-np.sin(Yangle),0,np.cos(Yangle)]],)
    Rz = np.array([[np.cos(Zangle),-np.sin(Zangle),0],
                   [np.sin(Zangle),np.cos(Zangle),0],
                   [0,0,1]])
    R = reduce( (lambda x,y:np.matmul(x,y)) , [Rz,Ry,Rx] )
    return R

def Rotation2DMatrix(Angle=0):
    """
    unit: degree [-180,180]
    """
    Angle = np.pi*Angle/180
        
    R = np.array([ [np.cos(Angle),-np.sin(Angle)],
                   [np.sin(Angle),np.cos(Angle)]])

    return R
  

class MultiObjectDataGenerator(object):
    """
    Input:
        # of data, outlierPortion
           
    Output:
        5D points (N,[X,Y,Z,x,y])
        Ground truth weight 
        Ground truth vec(A)
    """
    def __init__(self,inlier_face = None,inlier_angle=None):
        self.k = 0
        self.inlier_face = inlier_face
        self.inlier_angle = inlier_angle
        
    def set_Rt(self,R,T,Angle):
        self.R = R
        self.T = T
        self.Angle = Angle
        self.k = self.k+1
        


    def __getData(self,K,inlierPortion,numData=1000, pixel_std = 5,num_obj=1):        
        
        
        """
        we assume at most 3 objects appear
        x,y,z angle grid [0-40,40-80,80-120,120-160,160-200,200-240,240-280,280-320,320-360] 8 grids
        """
        
        # define output 
        
        R = np.zeros([3,3,3])
        T = np.zeros([3,3,1])
        Ps = np.zeros([numData,5])
        
        inlier = np.zeros(numData)
        
        # 3D points in camera coordinate system
        P2 =  np.ones([3,numData])
        P2[0:2,:] = np.random.uniform(-1,1,size=[2,numData])
        P2[2:3,:] = np.random.uniform(4,8,size=[1,numData])
        
        # image point
        P2_img = np.matmul(K,P2)
        P2_img = P2_img/P2_img[2:3,:]
        
        # add noise
        noise_img = np.zeros([3,numData])
        noise_img[0:2,:] = np.random.normal(0, pixel_std, [2,numData])
        P2_img_noise = P2_img + noise_img
        
        # back to normalized points
        P2_noise = np.matmul(np.linalg.inv(K),P2_img_noise) # [ 3 , numData ]
        
        # allocate the space for 3D points
        P1 = np.zeros([3,numData])

        # generate the inlier 3D points and the corresponding weight
        
        curr_idx = 0
        
        # randomly assign part of the object as inlier object
        if num_obj ==0:
            num_inlier = 0
        else:
            num_inlier = np.random.randint(0,2)
        
        for obj_idx in range(num_obj): 
            
            # number of inlier 3d-2d pairs
            obj_num = np.int(numData*inlierPortion[obj_idx])
            
            if obj_idx<=num_inlier:
                k = SampleAxis( bin_idx=self.inlier_face , FaceDict = FaceDict)
                theta = np.random.uniform(self.inlier_angle[0],self.inlier_angle[1],size=1)
                
                inlier[curr_idx:curr_idx+obj_num] = 1
            
            else:
                # uniform sampled on unit sphere 
                k = np.random.randn(3); 
                k = k/np.linalg.norm(k)
                theta = np.random.uniform(0,180,size=1) 
                
                # does not belong to the inlier bin               
                while DistMeasure(k,NorVec)==self.inlier_face and theta>=self.inlier_angle[0] and theta<self.inlier_angle[1]:
                    k = np.random.randn(3); 
                    k = k/np.linalg.norm(k)
                    theta = np.random.uniform(0,180,size=1)                     
                
                
            R_inlier = RotationMat(k,theta)
            
            # generate the 3D point in object coordinate system
            #R_inlier = Rotation3DMatrix(Angle[0],Angle[1],Angle[2])
            T_inlier = np.mean(P2[:,curr_idx:curr_idx+obj_num],axis=1,keepdims=True)
            P1[:,curr_idx:curr_idx+obj_num] = np.matmul(R_inlier.transpose(),(P2[:,curr_idx:curr_idx+obj_num]-T_inlier)) 
            
            if obj_idx<=num_inlier:
                R[obj_idx,:,:] = R_inlier
                T[obj_idx,:,:] = T_inlier
            
            # update the current index for next iteration
            curr_idx = curr_idx + obj_num

        
########## The rest are outliers ####################
        outlier_num = numData - curr_idx
        
        for i in range(outlier_num):
            #RandomAngle = np.random.uniform(low=0,high=360,size=3)
            #RandomR = Rotation3DMatrix(RandomAngle[0],RandomAngle[1],RandomAngle[2])   
            
            # uniformly sample a normalized axis over the unit sphere
            RandomK = np.random.randn(3)
            RandomK = RandomK/np.linalg.norm(RandomK)
            
            RandomTheta = np.random.uniform(0,180,size=1)
            RandomR = RotationMat(RandomK,RandomTheta)
            
            RandomT = np.random.uniform(-1,1,size=[3,1])
            RandomT[2:3,:] = np.random.uniform(4,8,size=[1,1])
            
           
            P1[:,curr_idx+i:curr_idx+i+1] = np.matmul(RandomR.transpose(),(P2[:,curr_idx+i:curr_idx+i+1]-RandomT))   
    
########## The rest are outliers ####################
        
        # form the output
        P2_noise = P2_noise[0:2,:]
        Ps = np.concatenate((P1,P2_noise),axis=0).transpose() #[numData,5]
        
        # shuffle the weight and Ps
        idx = np.arange(numData)
        np.random.shuffle(idx)
#        
        Ps = Ps[idx]        
        inlier = inlier[idx]

        # format Ps [numData,5] Weights[numData,]  e[1,12] 
        return [Ps ,  R, T , inlier]

    def BatchGetData(self, batch_size=10, numData=1000 ,  pixel_std=5 , max_inlierPortion = 0.3, seed=None , mode='train', test_inlierPortion=0.3,test_num_obj=2  ):
        
        # using the provided seed or a random seed to generate data
        if seed == None:
            seed = np.random.randint(0,1000,size=1)
        np.random.seed(seed)
 
        # define the output 
        batch_Ps = np.zeros([batch_size,1,numData,5])        
        batch_inlier = np.zeros([batch_size,numData])
        
        batch_R = np.zeros([batch_size,3,3,3])
        batch_T = np.zeros([batch_size,3,3,1])
        
        # define the camera intrinsic matrix
        u0 = 320
        v0 = 240
        fx = 800
        fy = 800
    
        K = np.array([[fx, 0., u0],[0., fy, v0],[0., 0., 1.]])
        
        # generate data
        
        min_inlierPortion = 0.2# minimum inlier portion
        
        for i in range(batch_size):            

            if mode == 'train': 
                # random portion of outliers for training
                num_obj = np.random.randint(1,4)
                inlierPortion = np.random.uniform(min_inlierPortion,max_inlierPortion,size = num_obj)
                
                
            elif mode == 'validate':
                # evenly sampled outlier portion
                num_obj = np.random.randint(1,4)
                inlierPortion =[min_inlierPortion + (max_inlierPortion-min_inlierPortion)*i/batch_size]*num_obj
                
                
            elif mode =='test':
                # fixed outlier portion
                num_obj = test_num_obj
                inlierPortion = [test_inlierPortion]*num_obj
                
            
            [ Ps, R , T , inlier] = self.__getData(K,inlierPortion,numData,pixel_std,num_obj)
            
            batch_Ps[i,0,:,:] = Ps
            batch_R[i,:,:,:] = R
            batch_T[i,:,:,:] = T 
            batch_inlier[i,:] = inlier     
            
        return [batch_Ps, batch_R,batch_T,batch_inlier]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    