"""Discriminator implementation.

This module defines the Discriminator to be used when constructing the Cycle GAN model.
The Discriminator makes use of the ConvolutionBlock's defined here.

"""

from collections import OrderedDict

import torch
from torch import nn


class ConvolutionBlock(nn.Module):
    """Performs a 2D convolution followed by an instance normalisation and a LeakyReLU activation on an image."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 instnorm: bool) -> None:
        """Initialises the convolutional block (convolutional layer, instance normalisation, and LeakyReLU) instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Desired number of output channels for the image.
            stride: Stride to be used for the convolution.
            instnorm: Boolean to decide whether to use instance normalisation or not.
        """
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels)
            if instnorm
            else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the convolution block using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the convolutional block.

        Returns:
            A torch tensor of the output of the convolutional block.

        """
        return self.block(x)


class Discriminator(nn.Module):
    """The Discriminator used in the CycleGAN."""
    def __init__(self,
                 in_channels: int,
                 inter_channels: int = 64) -> None:  # inter_channels = 64
        """Initialises the Discriminator.

        The Discriminator is made up with four ConvolutionBlock's and an output 2D convolutional layer.

        Args:
            in_channels: Number of channels in the input image.
            inter_channels: Number of channels (or filters) for output to be used with the first ConvolutionBlock.
              This value will be multiplied by factors of two in the intermediate ConvolutionBlock's before the output.
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(OrderedDict([
            ('C64', ConvolutionBlock(in_channels, inter_channels, stride=2, instnorm=False)),
            ('C128', ConvolutionBlock(inter_channels, inter_channels * 2, stride=2, instnorm=True)),
            ('C256', ConvolutionBlock(inter_channels * 2, inter_channels * 4, stride=2, instnorm=True)),
            ('C512', ConvolutionBlock(inter_channels * 4, inter_channels * 8, stride=1, instnorm=True)),
            ('out', nn.Conv2d(inter_channels * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        ]))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the Discriminator using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the Discriminator.

        Returns:
            A torch tensor of the output from the Discriminator.

        """
        return self.disc(x)
    