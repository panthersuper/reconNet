import torch.nn as nn

import sys, os
sys.path.append('model/')

from alexnet import alex_net
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class ReconNet(nn.Module):

    def __init__(self):
        super(ReconNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.latentV = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True), # latent vector, size:4096
        )

        # self.decoding = nn.Sequential(
        #     # decoding process
        #     # 2*2*2 de-conv3

        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),

        #     nn.ConvTranspose3d(512, 128, kernel_size=4, stride=4, padding=0), #8
        #     nn.BatchNorm3d(128, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(128, 32, kernel_size=4, stride=4, padding=0), #32
        #     nn.BatchNorm3d(32, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(32, 8, kernel_size=4, stride=4, padding=0), #128
        #     nn.BatchNorm3d(8, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(8, 1, kernel_size=2, stride=2, padding=0), #256
        #     nn.Tanh(),

        # )

        # self.decoding = nn.Sequential(
        #     # decoding process
        #     # 4/4/4 de-conv3

        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),

        #     UpsampleConv3Layer(64, 32, kernel_size=3, stride=1, upsample=2, outsize=8), #8
        #     nn.BatchNorm3d(32, affine=True),
        #     nn.ReLU(inplace=True), 

        #     UpsampleConv3Layer(32, 8, kernel_size=3, stride=1, upsample=4, outsize=32), #32
        #     nn.BatchNorm3d(8, affine=True),
        #     nn.ReLU(inplace=True), 

        #     UpsampleConv3Layer(8, 4, kernel_size=3, stride=1, upsample=4, outsize=128), #128
        #     nn.BatchNorm3d(4, affine=True),
        #     nn.ReLU(inplace=True), 

        #     UpsampleConv3Layer(4, 1, kernel_size=3, stride=1, upsample=2, outsize=256), #256
        #     nn.Tanh(),

        # )

        # self.decoding = nn.Sequential(
        #     # decoding process
        #     # 4/4/4 de-conv3

        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),

        #     nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2, padding=2), #8
        #     nn.BatchNorm3d(32, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(32, 16, kernel_size=6, stride=2, padding=2), #16
        #     nn.BatchNorm3d(16, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(16, 8, kernel_size=6, stride=2, padding=2), #32
        #     nn.BatchNorm3d(8, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(8, 4, kernel_size=6, stride=2, padding=2), #64
        #     nn.BatchNorm3d(4, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(4, 2, kernel_size=6, stride=2, padding=2), #128
        #     nn.BatchNorm3d(2, affine=True),
        #     nn.ReLU(inplace=True), 

        #     nn.ConvTranspose3d(2, 1, kernel_size=6, stride=2, padding=2), #256
        #     nn.Tanh(),

        # )

        self.decoding = nn.Sequential(
            # decoding process
            # 4/4/4 de-conv3 -> 32*32*32

            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64,64,kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2, padding=2), #8
            nn.BatchNorm3d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(32,32,kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(32, 8, kernel_size=6, stride=2, padding=2), #16
            nn.BatchNorm3d(8, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(8,8,kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(8, 2, kernel_size=6, stride=2, padding=2), #32     60*2*32*32*32
            nn.Tanh(),

        )

        # self.decoding = nn.Sequential(
        #     # decoding process
        #     # 4/4/4 de-conv3 -> 32*32*32

        #     nn.Linear(64 * 4 * 4 * 4, 32768*2),
        #     nn.Tanh(),

        # )

        self.softmax = nn.Sequential(

            nn.Softmax2d(),

        )

    

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.latentV(x)  # latent vector, size:4096

        # x = x.view(x.size(0),64,4,4,4) # reshape to 4/4/4 cube with 64 channels

        x = x.view(x.size(0), 128, 2, 2, 2)

        x = F.max_unpool3d(x, Variable(torch.Tensor(x.size()).zero_().long().cuda()), kernel_size=2, stride=2)
        deconv1 = nn.ConvTranspose3d(128, 128, 3, padding=1).cuda()
        x = deconv1(x)
        x = F.leaky_relu(x)

        x = F.max_unpool3d(x, Variable(torch.Tensor(x.size()).zero_().long().cuda()), kernel_size=2, stride=2)
        deconv1 = nn.ConvTranspose3d(128, 128, 3, padding=1).cuda()
        x = deconv1(x)
        x = F.leaky_relu(x)

        x = F.max_unpool3d(x, Variable(torch.Tensor(x.size()).zero_().long().cuda()), kernel_size=2, stride=2)
        deconv2 = nn.ConvTranspose3d(128, 64, 3, padding=1).cuda()
        x = deconv2(x)
        x = F.leaky_relu(x)

        x = F.max_unpool3d(x, Variable(torch.Tensor(x.size()).zero_().long().cuda()), kernel_size=2, stride=2)
        deconv3 = nn.ConvTranspose3d(64, 32, 3, padding=1).cuda()
        x = deconv3(x)
        x = F.leaky_relu(x)

        deconv4 = nn.ConvTranspose3d(32, 2, 3, padding=1).cuda()
        x = deconv4(x)

        # x = self.decoding(x) # convert to 3D voxel distribution
        x = x.view(batch_size,2,32,1024) # 60*2*32*1024  converted to 2d
        return x

class UpsampleConv3Layer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, outsize, upsample=None):
        super(UpsampleConv3Layer, self).__init__()
        self.upsample = upsample
        self.outsize = outsize
        # if upsample:
        #     self.upsample_layer = nn.functional.upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad3d(reflection_padding) #may modify to ReflectionPad
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride,padding=reflection_padding)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.upsample(x_in,scale_factor=self.upsample,size=self.outsize,mode='trilinear')

        # out = self.reflection_pad(x_in)
        out = self.conv3d(x_in)
        return out
