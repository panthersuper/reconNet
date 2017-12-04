import os
import numpy as np
import scipy.misc
# import h5py
np.random.seed(123)
import math
import os
from fnmatch import fnmatch
import scipy.io as spio

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.img_root = os.path.join(kwargs['img_root'])
        self.voxel_root = kwargs['voxel_root']
        self.voxel_size = kwargs['voxel_size']
        self.down_sample_scale = kwargs['down_sample_scale']

        # 'img_root': '../data/train_imgs/',   # MODIFY PATH ACCORDINGLY
        # 'voxel_root': '../data/train_voxels/',   # MODIFY PATH ACCORDINGLY
        # 'load_size': load_size,
        # 'fine_size': fine_size,
        # 'data_mean': data_mean,
        # 'randomize': True
        # 'voxel_size': voxel_size,


        # read data info from lists
        self.list_im = []
        self.list_vol = []

        #get data list
        for path, subdirs, files in os.walk(self.img_root):
            for name in files:
                if fnmatch(name, '*.png'):
                    img_dir =  os.path.join(path.split(self.img_root)[1],name)
                    img_dir = os.path.join(self.img_root,img_dir)
                    vol_dir = os.path.join(path.split(self.img_root)[1],'model.mat')
                    vol_dir = os.path.join(self.voxel_root,vol_dir)

                    self.list_im.append(img_dir)
                    self.list_vol.append(vol_dir)


        self.list_im = np.array(self.list_im, np.object)
        self.list_vol = np.array(self.list_vol, np.object)

        self.num = self.list_im.shape[0]

        print('# Images found:',self.num)

        # permutation
        perm = np.random.permutation(self.num)
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_vol[:, ...] = self.list_vol[perm, ...]

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 4))
        labels_batch = np.zeros((batch_size, self.voxel_size//self.down_sample_scale, self.voxel_size//self.down_sample_scale, self.voxel_size//self.down_sample_scale))
        dirs = []
        
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = math.floor((self.load_size-self.fine_size)/2)
                offset_w = math.floor((self.load_size-self.fine_size)/2)

            voxels = spio.loadmat(self.list_vol[self._idx], squeeze_me=True)["input"]
            voxels = np.asarray(voxels)

            # nvoxel = np.sum(np.reshape(voxels,(-1,256*256*256)))

            # print(nvoxel)
            # voxels = voxels/nvoxel*1000000
            dirs.append(self.list_vol[self._idx])

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = voxels[::self.down_sample_scale,::self.down_sample_scale,::self.down_sample_scale] # downsampling

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return images_batch, labels_batch


    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
