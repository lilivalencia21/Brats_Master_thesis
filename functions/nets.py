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