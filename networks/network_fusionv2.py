import torch

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianConv2d
from blitz.utils import variational_estimator
from torch.distributions.normal import Normal

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
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
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, bayesian_final: bool, dropout_rate: float = 0.1):
        super(Decoder, self).__init__()
        
        self.warn_ = False
        self.bayesian_final = bayesian_final
        self.num_layers = num_layers
        
        self.num_classes = 3
        
        self.upconvs = nn.ModuleList()
        
        in_channels = in_channels
        out_channels = out_channels
        
        for _ in range(num_layers):
            upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            
            self.upconvs.append(upconv)
            
            in_channels = out_channels
            out_channels = in_channels // 2
        
        
        if self.bayesian_final:
            if self.num_layers > 0:
                self.final = BayesianConv2d(self.upconvs[-1].out_channels, self.num_classes*2, kernel_size=(1,1), 
                                            prior_dist = Normal(0, 1),
                                            posterior_rho_init = -9.0)
            else:
                self.final = BayesianConv2d(in_channels, self.num_classes*2, kernel_size=(1,1))
        else:
            if self.num_layers > 0:
                self.dropout = nn.Dropout2d(dropout_rate)
                self.final = nn.Conv2d(self.upconvs[-1].out_channels, self.num_classes, kernel_size=1)
            else:
                self.dropout = nn.Dropout2d(dropout_rate)
                self.final = nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
    
    def forward(self, x):
        if self.num_layers == 0:
            x = self.dropout(x)
            x = self.final(x)
            return x
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
        x = self.dropout(x)
        x = self.final(x) 
        # currently, this output is N, 3, 96, 176. need to resize it to N, 3, 102, 180
        if x.shape != (x.shape[0], 3, 102, 180):
            if not self.warn_:
                print('WARNING: Resizing the output of the decoder using bilinear interpolation.')
                print('size:', x.shape)
            x = F.interpolate(x, size=(102, 180), mode='bilinear', align_corners=True)
            self.warn_ = True
        return x

class ClassAwareEdgeDetection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ClassAwareEdgeDetection, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.relu = nn.ReLU()

    #     # Initialize the weights and bias for edge detection
    #     self._init_weights()

    # def _init_weights(self):
    #     # Initialize weights with values that emphasize edges
    #     # Adjust these values based on experimentation
    #     edge_kernel = torch.tensor([[[-0.25, -0.25, -0.25],
    #                                 [-0.25,  0.5, -0.25],
    #                                 [-0.25, -0.25, -0.25]]], dtype=torch.float32)

    #     self.conv.weight = nn.Parameter(edge_kernel.repeat(self.out_channels, self.in_channels, 1, 1))
    #     nn.init.constant_(self.conv.bias, 0.0)  # Bias initialization

    def forward(self, x):
        edge_map = self.conv(x)
        edge_map = self.relu(edge_map)  # Ablate: Use a ReLU to ensure non-negative outputs
        return edge_map


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AttentionFusionBlock(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_3d, out_channels):
        super(AttentionFusionBlock, self).__init__()
        # Assuming in_channels_rgb == in_channels_3d for simplicity
        self.channel_attention = ChannelAttention(in_channels_rgb + in_channels_3d)
        self.spatial_attention = SpatialAttention()
        self.fusion_conv = nn.Conv2d(in_channels_rgb + in_channels_3d, out_channels, kernel_size=1)

    def forward(self, x_rgb, x_3d):
        '''
        x is already concatenated from the RGB and 3D encoders
        '''
        x = torch.cat([x_rgb, x_3d], dim=1)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.fusion_conv(x)
        return x

class AttentionFusionBlockWithEdgeLayer(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_3d, out_channels):
        super(AttentionFusionBlockWithEdgeLayer, self).__init__()
        
        # Class-aware edge detection layers for RGB and 3D features
        self.edge_detect_rgb = ClassAwareEdgeDetection(in_channels_rgb//2, in_channels_rgb//2)
        self.edge_detect_3d = ClassAwareEdgeDetection(in_channels_3d//2, in_channels_3d//2)

        # Channel and Spatial Attention
        self.channel_attention = ChannelAttention(in_channels_rgb + in_channels_3d)
        self.spatial_attention = SpatialAttention()
        
        # Fusion Convolution
        self.fusion_conv = nn.Conv2d(in_channels_rgb + in_channels_3d, out_channels, kernel_size=1)

    def forward(self, x_rgb, x_3d):
        # Apply edge detection to RGB and 3D features
        edge_rgb = self.edge_detect_rgb(x_rgb)
        edge_3d = self.edge_detect_3d(x_3d)

        # Combine edge information with original features
        x_rgb = torch.cat([x_rgb, edge_rgb], dim=1)
        x_3d = torch.cat([x_3d, edge_3d], dim=1)

        # Fusion of RGB and 3D data with edge information
        x = torch.cat([x_rgb, x_3d], dim=1)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.fusion_conv(x)
        return x
    
    
@variational_estimator
class BayesianFusionNetworkV2(nn.Module):
    def __init__(self, num_classes: int, dim_sources: List, is_BBB: bool = False):
        super(BayesianFusionNetworkV2, self).__init__()
        self.num_classes = num_classes
        self.dim_sources = dim_sources
        
        self.num_sources = len(dim_sources)
        
        self.is_BBB = is_BBB
        
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
            fusion_block = AttentionFusionBlock(in_channels_rgb = self.encoder.convs[i].conv.out_channels, 
                                                in_channels_3d = self.encoder.convs[i].conv.out_channels,
                                       out_channels = self.encoder.convs[i].conv.out_channels)
            self.fusion_blocks.append(fusion_block)
        
        # decoder
        self.decoder = Decoder(in_channels = self.encoder.convs[-1].conv.out_channels, out_channels = self.encoder.convs[-1].conv.in_channels, num_layers = 1, bayesian_final = self.is_BBB)
        
        
        
    def forward(self, x_rgb, x3d):
        x3d = torch.cat(x3d, dim=1)
        dnc = self.compress(x3d)
        features_rgb = self.encoder(x_rgb)
        features_dnc = self.encoder(dnc)
        
        # concatenate the features from the RGB and DnC encoders 
        # and apply the fusion blocks
        features = {}
        for i in range(len(features_rgb)):
            # features[f'x{i+1}'] = torch.cat([features_rgb[f'x{i+1}'], features_dnc[f'x{i+1}']], dim=1)
            features[f'x{i+1}'] = self.fusion_blocks[i](features_rgb[f'x{i+1}'] , features_dnc[f'x{i+1}'])
        
        # decoder
        out = self.decoder(features[f'x{len(features)}'])
        return out
    

class BayesianFusionNetworkV2EdgeLayer(nn.Module):
    def __init__(self, num_classes: int, dim_sources: List):
        super(BayesianFusionNetworkV2EdgeLayer, self).__init__()
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
            fusion_block = AttentionFusionBlockWithEdgeLayer(in_channels_rgb = 2 * self.encoder.convs[i].conv.out_channels, 
                                                in_channels_3d = 2 * self.encoder.convs[i].conv.out_channels,
                                       out_channels = self.encoder.convs[i].conv.out_channels)
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
        features = {}
        for i in range(len(features_rgb)):
            # features[f'x{i+1}'] = torch.cat([features_rgb[f'x{i+1}'], features_dnc[f'x{i+1}']], dim=1)
            features[f'x{i+1}'] = self.fusion_blocks[i](features_rgb[f'x{i+1}'] , features_dnc[f'x{i+1}'])
        
        # decoder
        out = self.decoder(features[f'x{len(features)}'])
        return out
    
        
if __name__ == '__main__':
    x_rgb = torch.randn(1, 3, 102, 180)
    x_depth = torch.randn(1, 1, 102, 180)
    x_normal = torch.randn(1, 3, 102, 180)
    x_mean_curvature = torch.randn(1, 1, 102, 180)
    x_gaussian_curvature = torch.randn(1, 1, 102, 180)
    model = BayesianFusionNetworkV2(num_classes=3, dim_sources=[x_rgb.shape[1], 
                                                                x_depth.shape[1], 
                                                                x_normal.shape[1],
                                                                x_mean_curvature.shape[1],
                                                                x_gaussian_curvature.shape[1]
                                                                ],
                                    is_BBB = False) 
    y = model(x_rgb, x3d = [x_depth, x_normal, x_mean_curvature, x_gaussian_curvature])
    
    print(y.shape)
    # print the nubmer of parameters and trainable parameters
    import torchsummary as summary
    summary.summary(model, input_size=[x_rgb.shape, [x_depth.shape, x_normal.shape, x_mean_curvature.shape, x_gaussian_curvature.shape]])