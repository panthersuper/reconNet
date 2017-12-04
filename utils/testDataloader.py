import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as s

load_size =128
fine_size = 113
data_mean = np.asarray([0.4,0.4,0.4,0])
batch_size = 20
voxel_size = 256

# Construct dataloader
opt_data_train = {
    'img_root': '../data/train_imgs/',   # MODIFY PATH ACCORDINGLY
    'voxel_root': '../data/train_voxels/',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'voxel_size': voxel_size,
    'data_mean': data_mean,
    'randomize': True,
    'down_sample_scale':1
}

loader_train = DataLoaderDisk(**opt_data_train)

data = loader_train.next_batch(batch_size)

dV = data[1]






print(np.sum(np.reshape(dV,(-1,256*256*256)),axis=1))
# print(data[1])

# print(dV.shape)