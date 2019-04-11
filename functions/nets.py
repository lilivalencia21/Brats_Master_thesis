import torch.nn.functional as F
import torch.nn as nn


class UNet3D(nn.Module):

    def __init__(self):
        super().__init__()

        kernel_size = 3
        self.down1 = nn.Conv3d(4, 8, kernel_size=3)
        self.down2 = nn.Conv3d(8, 16, kernel_size=3)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3)
        self.down4 = nn.Conv3d(32, 64, kernel_size=3)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=3)
        self.up2 = nn.ConvTranspose3d(32, 16, kernel_size=3)
        self.up3 = nn.ConvTranspose3d(16, 8, kernel_size=3)
        self.up4 = nn.ConvTranspose3d(8, 5, kernel_size=3)

    def forward(self, x):
        block1 = F.relu(self.down1(x))
        block2 = F.relu(self.down2(block1))
        block3 = F.relu(self.down3(block2))
        block4 = F.relu(self.down4(block3))

        block5 = F.relu(self.up1(block4))
        block6 = F.relu(self.up2(block5))
        block7 = F.relu(self.up3(block6))
        block8 = F.relu(self.up4(block7))

        x_out = F.softmax(block8, dim=1)

        return x_out

class UNet3DNNN(nn.Module):
    """ No New Net model"""

    def __init__(self):
        super().__init__()

        self.down1 = nn.Conv3d(4, 30, kernel_size=3)
        self.down2 = nn.Conv3d(30, 60, kernel_size=3)
        self.down3 = nn.Conv3d(60, 120, kernel_size=3)
        self.down4 = nn.Conv3d(120, 240, kernel_size=3)
        self.down5 = nn.Conv3d(240, 480, kernel_size=3)

        self.up1 = nn.ConvTranspose3d(480, 240, kernel_size=3)
        self.up2 = nn.ConvTranspose3d(240, 120, kernel_size=3)
        self.up3 = nn.ConvTranspose3d(120, 60, kernel_size=3)
        self.up4 = nn.ConvTranspose3d(60, 30, kernel_size=3)
        self.up5 = nn.ConvTranspose3d(30, 5, kernel_size=3)

    def forward(self, x):
        block1 = F.leaky_relu(self.down1(x))
        block2 = F.max_pool3d(F.leaky_relu(self.down2(block1)), (2, 2, 2))
        block3 = F.max_pool3d(F.leaky_relu(self.down3(block2)), (2, 2, 2))
        block4 = F.max_pool3d(F.leaky_relu(self.down4(block3)), (2, 2, 2))
        block5 = F.max_pool3d(F.leaky_relu(self.down5(block4)), (2, 2, 2))

        block6 = F.upsample(F.leaky_relu(self.up1(block5)), size=(2, 2, 2), mode='trilinear')
        block7 = F.upsample(F.leaky_relu(self.up2(block6)), size=(2, 2, 2), mode='trilinear')
        block8 = F.upsample(F.leaky_relu(self.up3(block7)), size=(2, 2, 2), mode='trilinear')
        block9 = F.upsample(F.leaky_relu(self.up4(block8)), size=(2, 2, 2), mode='trilinear')
        block10 = F.leaky_relu(self.up5(block9))

        x_out = F.softmax(block10, dim=1)

        return x_out