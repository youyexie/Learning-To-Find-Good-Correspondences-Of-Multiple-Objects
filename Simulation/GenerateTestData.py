# -*- coding: utf-8 -*-
"""
@author: Youye
"""
#%%
import numpy as np
from TestData import MultiObjectDataGenerator

# use GPU if it is available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper-parameters

# dataset parameters
seed = 3

numData = 200
test_inlierPortion = 0.3
test_num_obj = 3

test_size = 1000

# standard deviation of Gaussian noise unit: pixels
pixel_std = 1

inlier_face = None
inlier_angle = [0,180]

DataGenerator =  MultiObjectDataGenerator(inlier_face, inlier_angle)
#
### train set
[test_batch_Ps, test_batch_R,test_batch_T,test_batch_inlier] = DataGenerator.BatchGetData(test_size, numData,  pixel_std, seed = seed, mode='test',test_inlierPortion=test_inlierPortion,test_num_obj=test_num_obj )


#np.savez('Train', train_batch_Ps=train_batch_Ps,  train_batch_weight= train_batch_weight,train_batch_R=train_batch_R,train_batch_T=train_batch_T,train_batch_inlier=train_batch_inlier)
#
np.save('TestData\\Data_'+str(test_num_obj)+'object'+str(test_inlierPortion)+'std'+str(pixel_std)+'.npy',test_batch_Ps)
np.save('TestData\\Label_'+str(test_num_obj)+'object'+str(test_inlierPortion)+'std'+str(pixel_std)+'.npy',test_batch_inlier)

