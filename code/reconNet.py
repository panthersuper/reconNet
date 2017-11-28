import torch.nn as nn
import alexnet128.AlexNet as AlexNet

class ReconNet(nn.Module):

    def __init__(self):
        super(ReconNet, self).__init__()
        self.2dfeatures = nn.Sequential( # imput 113*113 image
            # encoding process
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


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

            UpsampleConv3Layer(512, 256, kernel_size=3, stride=1, upsample=2), # 4*4*4
            nn.InstanceNorm3d(256, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(256, 128, kernel_size=3, stride=1, upsample=2),  # 8*8*8
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(128, 64, kernel_size=3, stride=1, upsample=2),  # 16*16*16
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(64, 32, kernel_size=3, stride=1, upsample=2),  # 32*32*32
            nn.InstanceNorm3d(32, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(32, 16, kernel_size=3, stride=1, upsample=2),  # 64*64*64
            nn.InstanceNorm3d(16, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(16, 8, kernel_size=3, stride=1, upsample=2),  # 128*128*128
            nn.InstanceNorm3d(8, affine=True),
            nn.ReLU(inplace=True), 

            UpsampleConv3Layer(8, 1, kernel_size=3, stride=1, upsample=2),  # 256*256*256
            nn.InstanceNorm3d(1, affine=True),
            nn.ReLU(inplace=True), 
        )


    def forward(self, x):
        x = self.2dfeatures(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.latentV(x) # latent vector, size:4096
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
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad3d(reflection_padding) #may modify to ReflectionPad
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv3d(out)
        return out
