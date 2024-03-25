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


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
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
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout_rate: float = 0.1):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.upconvs = nn.ModuleList()
        
        in_channels = in_channels
        out_channels = out_channels
        
        for _ in range(num_layers):
            upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            
            self.upconvs.append(upconv)
            
            in_channels = out_channels
            out_channels = in_channels // 2
        if self.num_layers > 0:
            self.final = nn.Conv2d(self.upconvs[-1].out_channels, 3, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x):
        if self.num_layers == 0:
            x = self.dropout(x)
            x = self.final(x)
            return x
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
        # diff_X, diff_Y = x.size()[2] - self.final.kernel_size[0], x.size()[3] - self.final.kernel_size[1]
        # x = F.pad(x, (diff_X // 2, diff_X - diff_X // 2, diff_Y // 2, diff_Y - diff_Y // 2))
        x = self.dropout(x)
        x = self.final(x) 
        return x

class BayesianSegmentationNetwork(nn.Module):
    def __init__(self, num_classes: int):
        super(BayesianSegmentationNetwork, self).__init__()
        self.num_classes = num_classes
        
        self.encoder = Encoder(in_channels=3, out_channels=8, num_layers=2)

        self.decoder = Decoder(in_channels = self.encoder.convs[-1].conv.out_channels, out_channels = self.encoder.convs[-1].conv.in_channels, num_layers=1)
        
    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features['x2'])
        return x


        
        
if __name__ == '__main__':
    model = BayesianSegmentationNetwork(num_classes=3)
    x = torch.randn(1, 3, 102, 180)
    y = model(x)
    print(y.shape)
    print(y)