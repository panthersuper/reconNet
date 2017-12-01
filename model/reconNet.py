import torch.nn as nn

import sys, os
sys.path.append('model/')

from alexnet import alex_net

class ReconNet(nn.Module):

    def __init__(self):
        super(ReconNet, self).__init__()
        self.features = alex_net(pretrained=True)

        self.latentV = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), # latent vector, size:4096
        )


        self.decoding = nn.Sequential(
            # decoding process
            # 2*2*2 de-conv3

            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=4, padding=0), #8
            nn.BatchNorm3d(128, affine=True),
            nn.ReLU(inplace=True), 

            nn.ConvTranspose3d(128, 32, kernel_size=4, stride=4, padding=0), #32
            nn.BatchNorm3d(32, affine=True),
            nn.ReLU(inplace=True), 

            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=4, padding=0), #128
            nn.BatchNorm3d(8, affine=True),
            nn.ReLU(inplace=True), 

            nn.ConvTranspose3d(8, 1, kernel_size=2, stride=2, padding=0), #256
            nn.Tanh(),

        )


    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = self.latentV(x)  # latent vector, size:4096
        x = x.view(x.size(0),512,2,2,2) # reshape to 2 by 2 by 2 cube with 512 channels
        x = self.decoding(x) # convert to 3D voxel distribution

        return x


class UpsampleConv3Layer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConv3Layer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn._functions.thnn.UpsamplingNearest3d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad3d(reflection_padding) #may modify to ReflectionPad
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x.contiguous()
        if self.upsample:
            x_in = nn._functions.thnn.UpsamplingNearest3d.apply(x_in, [256,256,256], self.upsample)

        out = self.reflection_pad(x_in.contiguous())
        out = self.conv3d(out)
        return out
