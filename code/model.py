import torch as T
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.prelu = nn.PReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.pixshuff = nn.PixelShuffle(2)
        self.output = nn.Conv2d(out_channels, in_channels, kernel_size=9, padding=4, bias=False)

    def block(self):
        return nn.Sequential(
            self.conv2,
            self.batchnorm,
            self.prelu,
        )

    def input_block(self):
        return nn.Sequential(
            self.conv1,
            self.prelu,
        )

    def output_block(self):
        return nn.Sequential(
            self.conv3,
            self.pixshuff,
            self.prelu
        )

    def forward(self, x):
        block_in = self.input_block()(x)
        block1 = T.add(self.block()(block_in), block_in)
        block2 = T.add(self.block()(block1), block1)
        block3 = T.add(self.block()(block2), block2)
        block4 = T.add(self.block()(block3), block3)
        block_bridge = T.add(block4, block_in)
        block5 = self.output_block()(block_bridge)
        block6 = self.output_block()(block5)
        output = self.output(block6)
        return output


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels * 2)
        self.conv4 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels * 4)
        self.conv6 = nn.Conv2d(out_channels * 4, out_channels * 8, kernel_size=3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(out_channels * 8, out_channels * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(out_channels * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv8 = nn.Conv2d(out_channels * 8, out_channels * 16, kernel_size=1)
        self.batchnorm5 = nn.BatchNorm2d(out_channels * 16)
        self.out_conv = nn.Conv2d(out_channels * 16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def block(self):
        return nn.Sequential(
            self.conv2,
            self.batchnorm2,
            self.leakyrelu,
            self.conv3,
            self.batchnorm2,
            self.leakyrelu,
            self.conv4,
            self.batchnorm3,
            self.leakyrelu,
            self.conv5,
            self.batchnorm3,
            self.leakyrelu,
            self.conv6,
            self.batchnorm4,
            self.leakyrelu,
            self.conv7,
            self.batchnorm4,
            self.leakyrelu,
        )

    def input_block(self):
        return nn.Sequential(
            self.conv1,
            self.leakyrelu,
            self.conv1_1,
            self.batchnorm1,
            self.leakyrelu,
        )

    def out_block(self):
        return nn.Sequential(
            self.conv8,
            self.batchnorm5,
            self.leakyrelu,
            self.pool,
            self.out_conv,
            self.sigmoid
        )

    def forward(self, x):
        block_in = self.input_block()(x)
        x = self.block()(block_in)
        output = self.out_block()(x)
        return output