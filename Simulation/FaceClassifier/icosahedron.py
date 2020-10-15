# -*- coding: utf-8 -*-
"""
@author: Youye
"""
import numpy as np
from functools import reduce

cos = np.cos
sin = np.sin

d26 = np.deg2rad(26.57)
d36 = np.deg2rad(36)
d72 = np.deg2rad(72)

# vertice 12
vtop = np.array([0,0,1])

v1 = np.array([cos(d26),0,sin(d26)])
v2 = np.array([cos(d26)*cos(d36),cos(d26)*sin(d36),-sin(d26)])
v3 = np.array([cos(d26)*cos(d72),cos(d26)*sin(d72),sin(d26)])
v4 = np.array([-cos(d26)*cos(d72),cos(d26)*sin(d72),-sin(d26)])
v5 = np.array([-cos(d26)*cos(d36),cos(d26)*sin(d36),sin(d26)])
v6 = np.array([-cos(d26),0,-sin(d26)])
v7 = np.array([-cos(d26)*cos(d36),-cos(d26)*sin(d36),sin(d26)])
v8 = np.array([-cos(d26)*cos(d72),-cos(d26)*sin(d72),-sin(d26)])
v9 = np.array([cos(d26)*cos(d72),-cos(d26)*sin(d72),sin(d26)])
v10 =np.array([cos(d26)*cos(d36),-cos(d26)*sin(d36),-sin(d26)])

vbot = np.array([0,0,-1])

# normal vector of faces 20
n1 = np.reshape( vtop + v1 + v3 , [1,3] )
n2 = np.reshape( vtop + v3 + v5 , [1,3] )
n3 = np.reshape( vtop + v5 + v7 , [1,3] )
n4 = np.reshape( vtop + v7 + v9 , [1,3] )
n5 = np.reshape( vtop + v9 + v1 , [1,3] )

n6 = np.reshape( v1 + v2 + v3 , [1,3] )
n7 = np.reshape( v2 + v3 + v4 , [1,3] )
n8 = np.reshape( v3 + v4 + v5 , [1,3] )
n9 = np.reshape( v4 + v5 + v6 , [1,3] )
n10= np.reshape( v5 + v6 + v7 , [1,3] )
n11= np.reshape( v6 + v7 + v8 , [1,3] )
n12= np.reshape( v7 + v8 + v9 , [1,3] )
n13= np.reshape( v8 + v9 + v10 , [1,3] )
n14= np.reshape( v9 + v10 + v1 , [1,3] )
n15= np.reshape( v10 + v1 + v2 , [1,3] )

n16=np.reshape( vbot + v2 + v4 , [1,3] )
n17=np.reshape( vbot + v4 + v6 , [1,3] )
n18=np.reshape( vbot + v6 + v8 , [1,3] )
n19=np.reshape( vbot + v8 + v10 , [1,3] )
n20=np.reshape( vbot + v10 + v2 , [1,3] )

FaceDict = {}
FaceDict[0] = [vtop,v1,v3]
FaceDict[1] = [vtop,v3,v5]
FaceDict[2] = [vtop,v5,v7]
FaceDict[3] = [vtop,v7,v9]
FaceDict[4] = [vtop,v9,v1]
FaceDict[5] = [v1,v2,v3]
FaceDict[6] = [v2,v3,v4]
FaceDict[7] = [v3,v4,v5]
FaceDict[8] = [v4,v5,v6]
FaceDict[9] = [v5,v6,v7]
FaceDict[10] = [v6,v7,v8]
FaceDict[11] = [v7,v8,v9]
FaceDict[12] = [v8,v9,v10]
FaceDict[13] = [v9,v10,v1]
FaceDict[14] = [v10,v1,v2]
FaceDict[15] = [vbot,v2,v4]
FaceDict[16] = [vbot,v4,v6]
FaceDict[17] = [vbot,v6,v8]
FaceDict[18] = [vbot,v8,v10]
FaceDict[19] = [vbot,v10,v2]
np.savez('FaceDict',FaceDict=FaceDict)

# each row is a normal vector of a facet of the regular icosahedron
NorVec = np.concatenate([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20],axis=0)
NorVec = NorVec/np.linalg.norm(NorVec,axis=1,keepdims=True) #[20,3]
np.save('NorVec',NorVec)

def SampleAxis( bin_idx=None , FaceDict = FaceDict):
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


def findDeg(n1,n2):
    '''
    calculate the degree between n1 and n2
    '''
    cosd = np.sum(n1*n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
    return np.rad2deg(cosd)

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


