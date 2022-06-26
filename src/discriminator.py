import torch
from torch import nn
from collections import OrderedDict

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, instnorm):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels)
            if instnorm
            else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        # print("conv", self.block(x).size())
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, inter_channels=64):  # inter_channels = 64
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(OrderedDict([
            ('C64', ConvolutionBlock(in_channels, inter_channels, stride=2, instnorm=False)),  # 3 , 64
            ('C128', ConvolutionBlock(inter_channels, inter_channels * 2, stride=2, instnorm=True)),  # 64, 128
            ('C256', ConvolutionBlock(inter_channels * 2, inter_channels * 4, stride=2, instnorm=True)),  # 128, 256
            ('C512', ConvolutionBlock(inter_channels * 4, inter_channels * 8, stride=1, instnorm=True)),  # 256, 512
            ('out', nn.Conv2d(inter_channels * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        ]))

    def forward(self, x):
        return self.disc(x) # DO I NEED A SIGMOID HERE??

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

    rand_image2 = torch.rand(1, 3, 256, 256)
    discc = Discriminator(3)
    kk = discc(test_image)
    print(kk.size())

    # genny = Generator(3)
    # print("here")
    # print(test_image.size())
    # outss = genny(test_image)
