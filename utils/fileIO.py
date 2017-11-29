import scipy.io as spio
import numpy as np

params = {
    'voxel_path': "../data/train_voxels/039836/",
}

filedir = params['voxel_path'] + "model.mat"

mat = spio.loadmat(filedir, squeeze_me=True)["input"]
mat = np.asarray(mat)
print(mat.shape)
