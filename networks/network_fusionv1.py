import torch

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianConv2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# We now define the network fusion model BayesianFusionNetwork.
from typing import List

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = in_channels
        out_channels = out_channels
        
        for _ in range(num_layers):
            conv = ConvBlock(in_channels, out_channels)
            pool = nn.MaxPool2d(2, 2, return_indices=False)
            
            self.convs.append(conv)
            self.pools.append(pool)
            
            in_channels = out_channels
            out_channels *= 2

    def forward(self, x):
        features = {}

        for i in range(self.num_layers):
            x = self.convs[i](x)
            features[f'x{i+1}'] = x
            x = self.pools[i](x)
        
        return features


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.upconvs = nn.ModuleList()
        
        in_channels = in_channels
        out_channels = out_channels
        
        for _ in range(num_layers):
            upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            
            self.upconvs.append(upconv)
            
            in_channels = out_channels
            out_channels = in_channels // 2
                    
        self.final = nn.Conv2d(self.upconvs[-1].out_channels, 3, kernel_size=1)
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
        x = self.final(x) 
        # currently, this output is N, 3, 96, 176. we need to resize it to N, 3, 102, 180
        x = F.interpolate(x, size=(102, 180), mode='bilinear', align_corners=True)
        return x

class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(FusionBlock, self).__init__()
        
        self.conv  = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        

class BayesianFusionNetworkV1(nn.Module):
    def __init__(self, num_classes: int, dim_sources: List):
        super(BayesianFusionNetworkV1, self).__init__()
        self.num_classes = num_classes
        self.dim_sources = dim_sources
        
        self.num_sources = len(dim_sources)
        
        self.compress = nn.Conv2d(
            in_channels = sum(dim_sources) - dim_sources[0],
            out_channels = 3,
            kernel_size = 1,
        )
        
        self.encoder = Encoder(in_channels = 3, out_channels = 8, num_layers = 2)
        
        # Define Fusion Blocks for Each Intermediate Layer,
        # whose number is determined by the length of dict in Encoder
        
        self.fusion_blocks = nn.ModuleList()
        for i in range(len(self.encoder.convs)):
            # in_channels is twice the out_channels of the previous layer in the encoder
            # out_channels is the same as the out_channels of the previous layer in the encoder
            fusion_block = FusionBlock(in_channels = 2 * self.encoder.convs[i].conv.out_channels, 
                                       out_channels = self.encoder.convs[i].conv.out_channels, kernel_size = 1)
            self.fusion_blocks.append(fusion_block)
        
        # decoder -
        self.decoder = Decoder(in_channels = self.encoder.convs[-1].conv.out_channels, out_channels = self.encoder.convs[-1].conv.in_channels, num_layers = 1)
        
        
        
    def forward(self, x_rgb, x3d):
        x3d = torch.cat(x3d, dim=1)
        dnc = self.compress(x3d)
        features_rgb = self.encoder(x_rgb)
        features_dnc = self.encoder(dnc)
        
        # concatenate the features from the RGB and DnC encoders 
        # and apply the fusion blocks
        for i in range(len(features_rgb)):
            features_rgb[f'x{i+1}'] = torch.cat([features_rgb[f'x{i+1}'], features_dnc[f'x{i+1}']], dim=1)
            features_rgb[f'x{i+1}'] = self.fusion_blocks[i](features_rgb[f'x{i+1}'])
        
        # decoder
        out = self.decoder(features_rgb[f'x{len(features_rgb)}'])
        return out
    
        
if __name__ == '__main__':
    x_rgb = torch.randn(1, 3, 102, 180)
    x_depth = torch.randn(1, 1, 102, 180)
    x_normal = torch.randn(1, 3, 102, 180)
    x_mean_curvature = torch.randn(1, 1, 102, 180)
    x_gaussian_curvature = torch.randn(1, 1, 102, 180)
    model = BayesianFusionNetworkV1(num_classes=3, dim_sources=[x_rgb.shape[1], 
                                                                x_depth.shape[1], 
                                                                x_normal.shape[1]]) 
    y = model(x_rgb, x3d = [x_depth, x_normal])
    print(y.shape)
    # print the nubmer of parameters and trainable parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_trainable_params}')