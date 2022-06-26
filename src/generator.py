import torch
from torch import nn
from collections import OrderedDict


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("conv", self.block(x).size())
        return self.block(x)

class FractionalStridedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(FractionalStridedConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print("FractionalStridedConvBlock", self.block(x).size())
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        original_input = x.clone()
        k = original_input + self.residual(x)
        # print("Residual:", f"x size: {x.size()}", f"k size: {k.size()}")
        return original_input + self.residual(x)

class Generator(nn.Module):
    def __init__(self, in_channels, inter_channels=64):
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
            # in_channels, out_channels, kernel_size, stride)
        ]))

    def forward(self, x):
        X = self.gen(x)
        # print(torch.tanh(X).size())
        return torch.tanh(X)

if __name__ == "__main__":
    from datasets import ImageDataset
    import config
    import os
    from torchvision import transforms

    target_shape = 256
    tr = transforms.Compose([
        transforms.RandomCrop(target_shape),
        transforms.ToTensor()
    ])
    #
    path = os.path.join(config.DATA_DIR, "horse2zebra")
    dataset = ImageDataset(path=path, kind="test", transform=tr)
    test_image = dataset[0][0]
    # in_channels = dataset[0][0].size()[0]
    #
    # convblock = ConvolutionBlock(in_channels, 3, kernel_size=3, stride=2)
    # new_image = convblock.forward(test_image)
    # print(new_image.size())
    #
    # # resblock = ResidualBlock(new_image.size()[0], 3, kernel_size=1, stride=1)
    # # res = resblock.forward(new_image)
    # # print(res)
    #
    # inpprocess = InputOutputProcessing(in_channels, out_channels=64)
    # img_ready = inpprocess(test_image)
    # inprocess2 = InputOutputProcessing(64, 3)
    # img2 = inprocess2(img_ready)
    # print(img_ready.size())
    # print(img2.size())
    #
    # rand_image = torch.rand(256,256,256)
    # frac = FractionalStridedConvBlock(256, 3, 3, 2)
    # ki = frac.forward(rand_image)
    # print(ki.size())
    # from PIL import Image
    # import torchvision.transforms as transforms
    # tr = transforms.ToPILImage()
    # imgg = tr(ki)
    # imgg.show()

    rand_image2 = torch.rand(1, 3, 256, 256)
    genny = Generator(3)
    print("here")
    print(test_image.size())
    outss = genny(test_image)
    # import torchvision.transforms as transforms
    # tr = transforms.ToPILImage()
    # imgg = tr(outss)
    # imgg.show()
    # # test_image.show()
    #
    # imggorig = tr(test_image)
    # imggorig.show()


    # print(outss)
    # ResidualBlock(inter_channels * 4, inter_channels * 4, 3, 1)
    # inter_channels = 64
    # resblock_test = ResidualBlock(inter_channels * 4, inter_channels * 4, 3, 2)
    # img_test_torch = torch.rand(inter_channels * 4, 256, 256)
    # resblock_test(img_test_torch)

    # CHECK HOW THE SHAPE IS CHANGING FOR when I put something into the residual block
    # the residual convolutional layers should not change the shape/dimensions
    # currently dimensions are changed because of convolution layer so it cannot be added to the original image
    #
    #
    #