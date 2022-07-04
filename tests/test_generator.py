import os
import sys

import numpy as np

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

import generator as gen


def test_convolutionblock():
    input1 = torch.rand(3, 256, 256)
    convblock1 = gen.ConvolutionBlock(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
    feature_map1 = convblock1.forward(input1)

    input2 = torch.rand(1, 256, 256)
    convblock2 = gen.ConvolutionBlock(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    feature_map2 = convblock2.forward(input2)
    assert torch.Size([64, 256, 256]) == feature_map1.size()
    assert torch.Size([2, 254, 254]) == feature_map2.size()

def test_fractional_strided_conv_block():
    input1 = torch.rand(256, 64, 64)
    fracblock1 = gen.FractionalStridedConvBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2)
    feature_map1 = fracblock1.forward(input1)
    assert torch.Size([128, 128, 128]) == feature_map1.size()

def test_residual_block():
    input1 = torch.rand(256, 64, 64)
    resblock1 = gen.ResidualBlock(in_channels=256, out_channels=256)
    feature_map1 = resblock1.forward(input1)

    dim1 = np.random.randint(1, 3)
    dim2 = np.random.randint(1, 1000)
    dim3 = np.random.randint(1, 1000)
    input2 = torch.rand(dim1, dim2, dim3)
    resblock2 = gen.ResidualBlock(in_channels=dim1, out_channels=dim1)
    feature_map2 = resblock2.forward(input2)

    assert torch.Size([256, 64, 64]) == feature_map1.size()
    assert torch.Size([dim1, dim2, dim3]) == feature_map2.size()

def test_generator():
    input1 = torch.rand(3, 256, 256)
    generator1 = gen.Generator(in_channels=3, inter_channels=64)
    feature_map1 = generator1.forward(input1)
    assert torch.Size([3, 256, 256]) == feature_map1.size()
