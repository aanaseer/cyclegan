"""Generator implementation.

This module defines the Generator to be used when constructing the Cycle GAN model.
The Generator makes use of the ConvolutionBlock, FractionalStridedConvBlock, and ResidualBlock defined here.
"""

from collections import OrderedDict

import torch
from torch import nn


class ConvolutionBlock(nn.Module):
    """Performs a 2D convolution followed by an instance normalisation and a ReLU activation on an image."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int) -> None:
        """Initialises the convolutional block (convolutional layer, instance normalisation, and ReLU) instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Desired number of output channels for the image.
            kernel_size: Kernel size to use for the convolution.
            stride: Stride to be used for the convolution.
            padding: Padding size to use for the convolution.
        """
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the convolution block using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the convolutional block.

        Returns:
            A torch tensor of the output of the convolutional block.
        """
        return self.block(x)


class FractionalStridedConvBlock(nn.Module):
    """Performs a 2D transposed convolution followed by an instance normalisation and a ReLU activation on an image."""
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int) -> None:
        """Initialises the fractional strided convolutional block (convolutional layer, instance normalisation, and
        ReLU) instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Desired number of output channels for the image.
            kernel_size: Kernel size to use for the convolution.
            stride: Stride to be used for the convolution.
        """
        super(FractionalStridedConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the fractional strided convolutional block using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the convolutional block.

        Returns:
            A torch tensor of the output of the convolutional block.
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    """A residual block to be used in the Discriminator."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        """Initialises the residual block.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Desired number of output channels for the image.
        """
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the residual block using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the residual block.

        Returns:
            A torch tensor in which the original input image is added to the output of one forward pass through the
            ResidualBlock.
        """
        original_input = x.clone()
        k = original_input + self.residual(x)
        return original_input + self.residual(x)


class Generator(nn.Module):
    """The Generator used in CycleGAN."""
    def __init__(self,
                 in_channels: int,
                 inter_channels: int = 64) -> None:
        """Initialises the Generator.

        The Generator is made up with ConvolutionBlock's, ResidualBlock's, FractionalStridedConvBlock's and an
        output 2D convolutional layer.

        Args:
            in_channels: Number of channels in the input image.
            inter_channels: Number of channels (or filters) for output to be used with the first ConvolutionBlock.
              This value will be scaled in the intermediate layers before the output.
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(OrderedDict([
            ('c7s1-64', ConvolutionBlock(in_channels, inter_channels, 7, 1, 3)),  # 3, 64
            ('d128', ConvolutionBlock(inter_channels, inter_channels * 2, 3, 2, 1)),  # 64, 128
            ('d256', ConvolutionBlock(inter_channels * 2, inter_channels * 4, 3, 2, 1)),  # 128, 256
            ('R256-1', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-2', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-3', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-4', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-5', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-6', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-7', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-8', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('R256-9', ResidualBlock(inter_channels * 4, inter_channels * 4)),  # 256, 256
            ('u128', FractionalStridedConvBlock(inter_channels * 4, inter_channels * 2, 3, 2)),  # 256, 128
            ('u64', FractionalStridedConvBlock(inter_channels * 2, inter_channels, 3, 2)),  # 128, 64
            ('c7s1-3', nn.Conv2d(inter_channels, in_channels, 7, 1, 3))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the Generator using an input image, x.

        Args:
            x: A torch tensor of the image to be passed through the Generator.

        Returns:
            A torch tensor in which the original input image has been through one forward pass after which a non-linear
            activation of tanh is applied to it.
        """
        X = self.gen(x)
        # print(torch.tanh(X).size())
        return torch.tanh(X)

if __name__ == "__main__":
    # pass
    # from datasets import ImageDataset
    # import config
    # import os
    # from torchvision import transforms
    #
    # target_shape = 256
    # tr = transforms.Compose([
    #     transforms.RandomCrop(target_shape),
    #     transforms.ToTensor()
    # ])
    # #
    # path = os.path.join(config.DATA_DIR, "horse2zebra")
    # dataset = ImageDataset(path=path, kind="test", transform=tr)
    # test_image = dataset[0][0]
    #
    #
    #
    # # in_channels = dataset[0][0].size()[0]
    # #
    # convblock = ConvolutionBlock(3, 3, kernel_size=3, stride=2, padding=1)
    # new_image = convblock.forward(test_image)
    # print(type(new_image))
    # print("----")
    # inter_channels = 64
    # fake = torch.rand(inter_channels * 4, 256, 256)
    # fractional = FractionalStridedConvBlock(inter_channels * 4, inter_channels * 2, 3, 2)
    # print(type(fractional.forward(fake)))
    # # print(new_image.size())
    # #
    # resblock = ResidualBlock(new_image.size()[0], 3)
    # res = resblock.forward(new_image)
    # print("res")
    # print(res)
    # #
    # # inpprocess = InputOutputProcessing(in_channels, out_channels=64)
    # # img_ready = inpprocess(test_image)
    # # inprocess2 = InputOutputProcessing(64, 3)
    # # img2 = inprocess2(img_ready)
    # # print(img_ready.size())
    # # print(img2.size())
    # #
    # # rand_image = torch.rand(256,256,256)
    # # frac = FractionalStridedConvBlock(256, 3, 3, 2)
    # # ki = frac.forward(rand_image)
    # # print(ki.size())
    # # from PIL import Image
    # # import torchvision.transforms as transforms
    # # tr = transforms.ToPILImage()
    # # imgg = tr(ki)
    # # imgg.show()
    #
    # # rand_image2 =
    # # genny = Generator(3)
    # # genny.forward(test_image)
    # # print("here")
    # # print(test_image.size())
    # # outss = genny(test_image)
    # # import torchvision.transforms as transforms
    # # tr = transforms.ToPILImage()
    # # imgg = tr(outss)
    # # imgg.show()
    # # # test_image.show()
    # #
    # # imggorig = tr(test_image)
    # # imggorig.show()
    #
    #
    # # print(outss)
    # # ResidualBlock(inter_channels * 4, inter_channels * 4, 3, 1)
    # # inter_channels = 64
    # # resblock_test = ResidualBlock(inter_channels * 4, inter_channels * 4, 3, 2)
    # # img_test_torch = torch.rand(inter_channels * 4, 256, 256)
    # # resblock_test(img_test_torch)
    #
    # # CHECK HOW THE SHAPE IS CHANGING FOR when I put something into the residual block
    # # the residual convolutional layers should not change the shape/dimensions
    # # currently dimensions are changed because of convolution layer so it cannot be added to the original image
    # #
    # #
    # #