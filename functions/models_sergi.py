# Sergi Valverde 2019
# University of Girona
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResCoreElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 use_batchnorm=True):

                super(ResCoreElement, self).__init__()
                self.use_bn = use_batchnorm
                self.conv1 = nn.Conv3d(input_size,
                                       num_filters,
                                       kernel_size=3,
                                       padding=1)
                self.conv2 = nn.Conv3d(input_size,
                                       num_filters,
                                       kernel_size=1,
                                       padding=0)
                self.bn_add = nn.BatchNorm3d(num_filters)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_sum = self.bn_add(x_1 + x_2) if self.use_bn is True else x_1 + x_2
        return F.relu(x_sum)


class Pooling3D(nn.Module):
    """
    3D pooling layer by striding.
    """
    def __init__(self, input_size, use_batchnorm=True):
        super(Pooling3D, self).__init__()
        self.use_bn = use_batchnorm
        self.conv1 = nn.Conv3d(input_size,
                               input_size,
                               kernel_size=2,
                               stride=2)
        self.bn = nn.BatchNorm3d(input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x) if self.use_bn is True else x
        return F.relu(x)


class ResUnet(nn.Module):
    """
    Basic U-net model using residual layers. Control the input channels,
    output channels, and scaling of output filters
    """
    def __init__(self, input_channels,
                 output_channels,
                 scale,
                 classification=True,
                 use_bn=True):
        super(ResUnet, self).__init__()
        self.use_bn = use_bn
        self.classification = classification
        # conv 1 down
        self.conv1 = ResCoreElement(input_channels, int(scale * 32), use_bn)
        self.pool1 = Pooling3D(int(scale * 32))

        # conv 2 down
        self.conv2 = ResCoreElement(int(scale * 32), int(scale * 64), use_bn)
        self.pool2 = Pooling3D(int(scale * 64))

        # conv 3 down
        self.conv3 = ResCoreElement(int(scale * 64), int(scale * 128), use_bn)
        self.pool3 = Pooling3D(int(scale * 128))

        # conv 4
        self.conv4 = ResCoreElement(int(scale*128), int(scale*256), use_bn)

        # up 1
        self.up1 = nn.ConvTranspose3d(int(scale*256),
                                      int(scale*128),
                                      kernel_size=2,
                                      stride=2)
        # conv 5
        self.conv5 = ResCoreElement(int(scale*128), int(scale*128), use_bn)

        # conv 6 up
        self.bn_add35 = nn.BatchNorm3d(int(scale*128))
        self.conv6 = ResCoreElement(int(scale*128), int(scale*128), use_bn)
        self.up2 = nn.ConvTranspose3d(int(scale*128),
                                      int(scale*64),
                                      kernel_size=2,
                                      stride=2)

        # conv 7 up
        self.bn_add22 = nn.BatchNorm3d(int(scale*64))
        self.conv7 = ResCoreElement(int(scale*64), int(scale*64), use_bn)
        self.up3 = nn.ConvTranspose3d(int(scale*64),
                                      int(scale*32),
                                      kernel_size=2,
                                      stride=2)

        # conv 8 up
        self.bn_add13 = nn.BatchNorm3d(int(scale*32))
        self.conv8 = ResCoreElement(int(scale*32), int(scale*32), use_bn)

        # reconstruction
        self.conv9 = nn.Conv3d(int(scale * 32),
                               output_channels,
                               kernel_size=1)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("ResUnet3D network with {} parameters".format(nparams))

    def forward(self, x, encoder=False):
        # --------------------
        # encoder
        # --------------------
        x1 = self.conv1(x)
        x1d = self.pool1(x1)
        x2 = self.conv2(x1d)
        x2d = self.pool2(x2)
        x3 = self.conv3(x2d)
        x3d = self.pool3(x3)
        x4 = self.conv4(x3d)

        # --------------------
        # decoder
        # --------------------
        up1 = self.up1(x4)
        x5 = self.conv5(up1)

        add_35 = self.bn_add35(x5 + x3) if self.use_bn is True else x5 + x3
        x6 = self.conv6(add_35)
        up2 = self.up2(x6)

        add_22 = self.bn_add22(up2 + x2) if self.use_bn is True else x2 + up2
        x7 = self.conv7(add_22)
        up3 = self.up3(x7)

        add_13 = self.bn_add13(up3 + x1) if self.use_bn is True else x1 + up3
        x8 = self.conv8(add_13)

        # return only the encoder part if selected
        if encoder is True:
            return x1, x2, x3, x4
        else:
            # return prediction is selected
            return F.softmax(self.conv9(x8), dim=1) if self.classification \
                else self.conv9(x8)


class Unet3D(nn.Module):
    def __init__(self,
                 in_ch=1,
                 out_ch=1,
                 ndims=3,
                 nfilts = 32,
                 d_prob=0.5,
                 softmax_out=True):
        super(Unet3D, self).__init__()

        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d

        self.conv1 = torch.nn.Sequential(
            Conv(in_ch, nfilts, 3, padding=1),
            BatchNorm(nfilts, momentum=0.01),
            nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            Conv(nfilts, 2 * nfilts, 3, padding=1),
            BatchNorm(2 * nfilts, momentum=0.01),
            nn.ReLU())
        self.down1 = MaxPool(2)

        self.conv3 = torch.nn.Sequential(
            Conv(2 * nfilts, 2 * nfilts, 3, padding=1),
            BatchNorm(2 * nfilts, momentum=0.01),
            nn.ReLU())
        self.conv4 = torch.nn.Sequential(
            Conv(2 * nfilts, 4 * nfilts, 3, padding=1),
            BatchNorm(4 * nfilts, momentum=0.01),
            nn.ReLU())
        self.down2 = MaxPool(2)

        self.conv5 = torch.nn.Sequential(
            Conv(4 * nfilts, 4 * nfilts, 3, padding=1),
            BatchNorm(4 * nfilts, momentum=0.01),
            nn.ReLU())
        self.conv6 = torch.nn.Sequential(
            Conv(4 * nfilts, 8 * nfilts, 3, padding=1),
            BatchNorm(8 * nfilts, momentum=0.01),
            nn.ReLU())
        self.down3 = MaxPool(2)

        self.conv7 = torch.nn.Sequential(
            Conv(8 * nfilts, 8 * nfilts, 3, padding=1),
            BatchNorm(8 * nfilts, momentum=0.01),
            nn.ReLU())
        self.conv8 = torch.nn.Sequential(
            Conv(8 * nfilts, 16 * nfilts, 3, padding=1),
            BatchNorm(16 * nfilts, momentum=0.01),
            nn.ReLU())
        self.upconv1 = ConvTranspose(16 * nfilts, 16 * nfilts, 3,
                                     padding=1,
                                     output_padding=1,
                                     stride=2)

        self.conv9 = torch.nn.Sequential(
            Conv(16 * nfilts + 8 * nfilts, 8 * nfilts, 3, padding=1),
            BatchNorm(8 * nfilts, momentum=0.01),
            nn.ReLU())
        self.conv10 = torch.nn.Sequential(
            Conv(8 * nfilts, 8 * nfilts, 3, padding=1),
            BatchNorm(8 * nfilts, momentum=0.01),
            nn.ReLU())
        self.upconv2 = ConvTranspose(8 * nfilts,
                                     8 * nfilts, 3,
                                     padding=1,
                                     output_padding=1,
                                     stride=2)

        self.conv11 = torch.nn.Sequential(
            Conv(8 * nfilts + 4 * nfilts, 4 * nfilts, 3, padding=1),
            BatchNorm(4 * nfilts, momentum=0.01),
            nn.ReLU())
        self.conv12 = torch.nn.Sequential(
            Conv(4 * nfilts, 4 * nfilts, 3, padding=1),
            BatchNorm(4 * nfilts, momentum=0.01),
            nn.ReLU())
        self.upconv3 = ConvTranspose(4 * nfilts, 4 * nfilts, 3,
                                     padding=1,
                                     output_padding=1,
                                     stride=2)

        self.conv13 = torch.nn.Sequential(Conv(4 * nfilts + 2 * nfilts, 2 * nfilts, 3, padding=1),
                                          BatchNorm(2 * nfilts, momentum=0.01),
                                          nn.ReLU())
        self.conv14 = torch.nn.Sequential(Conv(2 * nfilts, 2 * nfilts, 3, padding=1),
                                          BatchNorm(2 * nfilts, momentum=0.01),
                                          nn.ReLU())

        self.out_conv = Conv(2 * nfilts, out_ch, 3, padding=1)
        self.softmax_out = nn.Softmax(dim=1) if softmax_out is True else None

        self.d5 = nn.AlphaDropout(p=d_prob)
        self.d6 = nn.AlphaDropout(p=d_prob)
        self.d7 = nn.AlphaDropout(p=d_prob)
        self.d8 = nn.AlphaDropout(p=d_prob)
        self.d9 = nn.AlphaDropout(p=d_prob)
        self.d10 = nn.AlphaDropout(p=d_prob)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("Unet3D (Cicek) network with {} parameters".format(nparams))

    def forward(self, x_in):
        l1_start = self.conv1(x_in)
        l1_end = self.conv2(l1_start)

        l2_start = self.down1(l1_end)
        l2_mid = self.conv3(l2_start)
        l2_end = self.conv4(l2_mid)

        l3_start = self.down2(l2_end)
        l3_mid = self.d5(self.conv5(l3_start))
        l3_end = self.d6(self.conv6(l3_mid))

        l4_start = self.down3(l3_end)
        l4_mid = self.d7(self.conv7(l4_start))
        l4_end = self.d8(self.conv8(l4_mid))

        r3_start = torch.cat((l3_end, self.upconv1(l4_end)), dim=1)
        r3_mid = self.d9(self.conv9(r3_start))
        r3_end = self.d10(self.conv10(r3_mid))

        r2_start = torch.cat((l2_end, self.upconv2(r3_end)), dim=1)
        r2_mid = self.conv11(r2_start)
        r2_end = self.conv12(r2_mid)

        r1_start = torch.cat((l1_end, self.upconv3(r2_end)), dim=1)
        r1_mid = self.conv13(r1_start)
        r1_end = self.conv14(r1_mid)

        x_out = self.out_conv(r1_end)
        if self.softmax_out is not None:
            x_out = self.softmax_out(x_out)

        return x_out