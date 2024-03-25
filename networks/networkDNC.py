'''
Architecture Definition:


'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet
import torch.nn.functional as F

# GPU UTILIZATION
import GPUtil

from getPCA import getPCA


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGGNet(VGG):
    def __init__(self, pretrained = True, model = 'vgg16', requires_grad = True, remove_fc = True, show_params = False):
        super().__init__(make_layers(cfg[model], batch_norm=False))
        self.ranges = ranges[model]
        
        if pretrained:
            self.load_state_dict(models.vgg16(pretrained=True).state_dict())
        

        # nn.Sequential([nn.Conv2d, models])
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            del self.classifier
        
        if show_params:
            for name,param in self.named_parameters():
                print(name, param.size())
        
    
    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
        
        

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out


class FCNDepth(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.compress = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=1)
        self.pretrained_net = pretrained_net
        # self.aspp = ASPP(in_channels=512, out_channels=512, dilation_rates = [6,12,18])
        # This one is for NL-Fusion w/o attention vector
        self.fusionConv = nn.ModuleDict({'x5': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),  
                           'x4': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                           'x3': nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                           'x2': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                           'x1': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)})
        # self.fusionConv = nn.Conv2d(1024, 512, 1)

        # # This one is for NL-Fusion w/ attention vector
        self.fsp_rgb = {'x5': FSP(in_planes=512, out_planes=512, reduction = 16),
                        'x4': FSP(in_planes=512, out_planes=512, reduction = 16),
                        'x3': FSP(in_planes=256, out_planes=256, reduction = 16),
                        'x2': FSP(in_planes=128, out_planes=128, reduction = 16),
                        'x1': FSP(in_planes=64, out_planes=64, reduction = 16)}

        self.fsp_hha = {'x5': FSP(in_planes=512, out_planes=512, reduction = 16),
                        'x4': FSP(in_planes=512, out_planes=512, reduction = 16),
                        'x3': FSP(in_planes=256, out_planes=256, reduction = 16),
                        'x2': FSP(in_planes=128, out_planes=128, reduction = 16),
                        'x1': FSP(in_planes=64, out_planes=64, reduction = 16)}
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x_rgb, x_depth, x_normal, x_curvature):
        dnc = self.compress(torch.cat([x_depth,x_normal,x_curvature],1))
        output_rgb = self.pretrained_net(x_rgb)
        output_depth = self.pretrained_net(dnc)
        
        x5_rgb = output_rgb['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_rgb = output_rgb['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_rgb = output_rgb['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_rgb = output_rgb['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_rgb = output_rgb['x1']  # size=(N, 64, x.H/2,  x.W/2)

        x5_depth = output_depth['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4_depth = output_depth['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3_depth = output_depth['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2_depth = output_depth['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1_depth = output_depth['x1']  # size=(N, 64, x.H/2,  x.W/2)

        
        # FUSION METHOD : SIMPLE ELEMENT-WISE ADDITION
        # x5_eladd = x5_rgb + x5_depth
        # x4_eladd = x4_rgb + x4_depth
        # x3_eladd = x3_rgb + x3_depth
        # x2_eladd = x2_rgb + x2_depth
        # x1_eladd = x1_rgb + x1_depth
        

        # # # Fusion Block : NON-LINEAR WEIGHTED COMBINATION 
        # x5_nlcombo = self.relu(self.fuseBlock_1(x5_rgb, x5_depth, 'x5'))
        # x4_nlcombo = self.relu(self.fuseBlock_1(x4_rgb, x4_depth, 'x4'))
        # x3_nlcombo = self.relu(self.fuseBlock_1(x3_rgb, x3_depth, 'x3'))
        # x2_nlcombo = self.relu(self.fuseBlock_1(x2_rgb, x2_depth, 'x2'))
        # x1_nlcombo = self.relu(self.fuseBlock_1(x1_rgb, x1_depth, 'x1'))

        # Fusion Block: NON-LINEAR WEIGHTED COMBINATION W/ RECALIBRATION FILTERING
        x5 = self.relu(self.fuseBlock_2(x5_rgb, x5_depth, 'x5'))
        x4 = self.relu(self.fuseBlock_2(x4_rgb, x4_depth, 'x4'))
        x3 = self.relu(self.fuseBlock_2(x3_rgb, x3_depth, 'x3'))
        x2 = self.relu(self.fuseBlock_2(x2_rgb, x2_depth, 'x2'))
        x1 = self.relu(self.fuseBlock_2(x1_rgb, x1_depth, 'x1'))


        # Fusion Block: SA-Gate
        

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                              # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
    def fuseBlock_1(self, rgb, depth, layer):
        '''
        Fusion Block 1: 
        Weighted mean of RGB and Depth using 1x1 convolution 
        '''
        fused = torch.cat((rgb, depth), dim=1)
        # weight = F.softmax(self.fusionConv[layer].weight, dim=1)
        x = self.fusionConv[layer](fused)
        # x = F.conv2d(fused, weight)

        return x
    

    def fuseBlock_2(self, rgb, depth, layer):
        '''
        Fusion Block 2: 
        Weighted mean of RGB and Depth using 1x1 convolution 
        Pre-processing using Depth and RGB Recalibration 
        '''
        N,b,_,_ = rgb.shape
        rec_rgb = self.fsp_rgb[layer](depth, rgb)
        rec_depth = self.fsp_hha[layer](rgb, depth)
        cat_fea = torch.cat([rec_rgb, rec_depth], dim=1)
        # weight = F.softmax(self.fusionConv[layer].weight, dim=1)
        # x = F.conv2d(cat_fea, weight)
        x = self.fusionConv[layer](cat_fea)

        return x
        

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        ).to(device)

        self.out_planes = out_planes
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCNDepth(pretrained_net=vgg_model, n_class=3)
