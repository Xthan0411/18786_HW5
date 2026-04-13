# CMU CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # up_conv1: 将 1x1 的 noise 向量直接通过卷积扩展为 4x4 特征图
        # 输入 noise_size x 1 x 1 -> 256 x 4 x 4
        # 由于输入空间尺寸为 1x1, 这里不使用 Upsample, 而是直接用 Conv2d
        # kernel=4, stride=1, padding=3 时: out = 1 + 2*3 - 4 + 1 = 4
        self.up_conv1 = conv(
            in_channels=noise_size, out_channels=conv_dim * 8,
            kernel_size=4, stride=1, padding=3,
            norm='instance', activ='relu'
        )
        # up_conv2: 4x4 -> 8x8, 通道 256 -> 128
        self.up_conv2 = up_conv(
            in_channels=conv_dim * 8, out_channels=conv_dim * 4,
            kernel_size=3, stride=1, padding=1,
            scale_factor=2, norm='instance', activ='relu'
        )
        # up_conv3: 8x8 -> 16x16, 通道 128 -> 64
        self.up_conv3 = up_conv(
            in_channels=conv_dim * 4, out_channels=conv_dim * 2,
            kernel_size=3, stride=1, padding=1,
            scale_factor=2, norm='instance', activ='relu'
        )
        # up_conv4: 16x16 -> 32x32, 通道 64 -> 32
        self.up_conv4 = up_conv(
            in_channels=conv_dim * 2, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1,
            scale_factor=2, norm='instance', activ='relu'
        )
        # up_conv5: 32x32 -> 64x64, 通道 32 -> 3 (RGB)
        # 最后一层不使用 normalization, 使用 tanh 将输出限制到 [-1, 1]
        self.up_conv5 = up_conv(
            in_channels=conv_dim, out_channels=3,
            kernel_size=3, stride=1, padding=1,
            scale_factor=2, norm=None, activ='tanh'
        )

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        # 按顺序将 noise 依次通过 5 层上采样卷积块, 逐步放大到 64x64 的图像
        out = self.up_conv1(z)   # -> B x 256 x 4 x 4
        out = self.up_conv2(out)  # -> B x 128 x 8 x 8
        out = self.up_conv3(out)  # -> B x 64 x 16 x 16
        out = self.up_conv4(out)  # -> B x 32 x 32 x 32
        out = self.up_conv5(out)  # -> B x 3 x 64 x 64
        return out


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        # conv1: 3x64x64 -> 32x32x32 (已提供)
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')
        # conv2: 32x32x32 -> 64x16x16, kernel=4, stride=2, padding=1 下采样 2 倍
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')
        # conv3: 64x16x16 -> 128x8x8
        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu')
        # conv4: 128x8x8 -> 256x4x4
        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')
        # conv5: 256x4x4 -> 1x1x1, 输出判别器的 logit
        # kernel=4, stride=1, padding=0 时 4x4 卷积正好得到 1x1, 且不使用 norm/激活
        self.conv5 = conv(256, 1, 4, 1, 0, norm=None, activ=None)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
