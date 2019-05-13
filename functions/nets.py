import torch.nn.functional as F
import torch.nn as nn
import torch


class UNet3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = nn.Conv3d(4, 32, kernel_size=3, padding=1)
        self.down2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.down3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.down4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose3d(256, 64, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose3d(128, 32, kernel_size=3, padding=1)

        self.out = nn.Conv3d(64, 4, kernel_size=1)

    def forward(self, x):
        block1 = F.relu(self.down1(x))
        block2 = F.relu(self.down2(block1))
        block3 = F.relu(self.down3(block2))
        block4 = F.relu(self.down4(block3))

        block5 = F.relu(self.up1(block4))
        skipCon1 = torch.cat([block5, block3], dim=1)
        block6 = F.relu(self.up2(skipCon1))
        skipCon2 = torch.cat([block6, block2], dim=1)
        block7 = F.relu(self.up3(skipCon2))
        skipCon3 = torch.cat([block7, block1], dim=1)

        block8 = self.out(skipCon3)

        x_out = F.softmax(block8, dim=1)

        return x_out

class UNet3DNNN(nn.Module):
    """ No New Net model"""

    def __init__(self):
        super().__init__()

        self.down1 = nn.Conv3d(4, 30, kernel_size=3, padding=1)
        self.down2 = nn.Conv3d(30, 60, kernel_size=3, padding=1)
        self.down3 = nn.Conv3d(60, 120, kernel_size=3, padding=1)
        self.down4 = nn.Conv3d(120, 240, kernel_size=3, padding=1)
        self.down5 = nn.Conv3d(240, 480, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose3d(480, 240, kernel_size=3, padding=1)
        # same number of filters in up2 is the same given the skip connection
        self.up2 = nn.ConvTranspose3d(480, 120, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose3d(240, 60, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose3d(120, 30, kernel_size=3, padding=1)

        self.out = nn.Conv3d(60, 4, kernel_size=1)

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            block1 = F.relu(self.down1(x))
            block2 = F.max_pool3d(F.relu(self.down2(block1)), (2, 2, 2))
            block3 = F.max_pool3d(F.relu(self.down3(block2)), (2, 2, 2))
            block4 = F.max_pool3d(F.relu(self.down4(block3)), (2, 2, 2))
            block5 = F.max_pool3d(F.relu(self.down5(block4)), (2, 2, 2))

            block6 = F.interpolate(F.relu(self.up1(block5)), size=block4.shape[-3:], mode='trilinear')
            skipCon1 = torch.cat([block4, block6], dim=1)
            block7 = F.interpolate(F.relu(self.up2(skipCon1)), size=block3.shape[-3:], mode='trilinear')
            skipCon2 = torch.cat([block3, block7], dim=1)
            block8 = F.interpolate(F.relu(self.up3(skipCon2)), size=block2.shape[-3:], mode='trilinear')
            skipCon3 = torch.cat([block2, block8],  dim=1)
            block9 = F.interpolate(F.relu(self.up4(skipCon3)), size=block1.shape[-3:], mode='trilinear')
            skipCon4 = torch.cat([block1, block9], dim=1)

            block10 = self.out(skipCon4)

            x_out = F.softmax(block10, dim=1)

        return x_out