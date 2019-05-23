import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class UNet3D(nn.Module):

    def __init__(self,  nfilts=4):
        super().__init__()

        self.down1 = nn.Conv3d(4, nfilts, kernel_size=3, padding=1)
        self.down2 = nn.Conv3d(nfilts, nfilts*2, kernel_size=3, padding=1)
        self.down3 = nn.Conv3d(nfilts*2, nfilts*4, kernel_size=3, padding=1)
        self.down4 = nn.Conv3d(nfilts*4, nfilts*8, kernel_size=3, padding=1)

        self.middle = nn.Conv3d(nfilts*8, nfilts*16, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose3d(nfilts*16,  nfilts*8, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose3d(nfilts*16, nfilts*4, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose3d(nfilts*8, nfilts*2, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose3d(nfilts*4, nfilts, kernel_size=3, padding=1)

        self.out = nn.Conv3d(nfilts*2, 4, kernel_size=1)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("UNet3D network with {} parameters".format(nparams))

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            block1 = F.relu(self.down1(x))
            pooling1 = F.max_pool3d(block1, (2, 2, 2))
            block2 = F.relu(self.down2(pooling1))
            pooling2 = F.max_pool3d(block2, (2, 2, 2))
            block3 = F.relu(self.down3(pooling2))
            pooling3 = F.max_pool3d(block3, (2, 2, 2))
            block4 = F.relu(self.down4(pooling3))
            pooling4 = F.max_pool3d(block4, (2, 2, 2))

            block5 = F.relu(self.middle(pooling4))

            block6 = F.interpolate(F.relu(self.up1(block5)), size=block4.shape[2:], mode='trilinear')
            skipCon1 = torch.cat([block6, block4], dim=1)
            block7 =  F.interpolate(F.relu(self.up2(skipCon1)), size=block3.shape[2:], mode='trilinear')
            skipCon2 = torch.cat([block7, block3], dim=1)
            block8 = F.interpolate(F.relu(self.up3(skipCon2)), size=block2.shape[2:], mode='trilinear')
            skipCon3 = torch.cat([block8, block2], dim=1)
            block9 =  F.interpolate(F.relu(self.up4(skipCon3)), size=block1.shape[2:], mode='trilinear')
            skipCon4 =  torch.cat([block9, block1], dim=1)

            block10 = self.out(skipCon4)

            x_out = F.softmax(block10, dim=1)

        return x_out

