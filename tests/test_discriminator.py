import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

import discriminator as disc

def test_convolutionblock():
    input1 = torch.rand(3, 256, 256)
    convblock1 = disc.ConvolutionBlock(in_channels=3, out_channels=64, stride=2, instnorm=False)
    feature_map1 = convblock1.forward(input1)

    convblock2 = disc.ConvolutionBlock(in_channels=64, out_channels=128, stride=2, instnorm=True)
    feature_map2 = convblock2.forward(feature_map1)

    assert torch.Size([64, 128, 128]) == feature_map1.size()
    assert torch.Size([128, 64, 64]) == feature_map2.size()

def test_discriminator():
    input1 = torch.rand(3, 256, 256)
    disc1 = disc.Discriminator(in_channels=3, inter_channels=64)
    feature_map1 = disc1.forward(input1)

    assert torch.Size([1, 30, 30]) == feature_map1.size()