import torchvision
import torch
from torch import nn
from torch.nn import init
from models.utils import pooling

class GEN(nn.Module):
    def __init__(self, in_feat_dim, out_img_dim, config, **kwargs):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.out_img_dim = out_img_dim

        self.conv0 = nn.Conv2d(self.in_feat_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv4 = nn.Conv2d(32, self.out_img_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)

        self.up = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(64)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        x = self.conv0(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.up(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv4(x)
        x = torch.tanh(x)

        return x



        

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
        self.uncloth_dim = config.MODEL.NO_CLOTHES_DIM//2
        self.contour_dim = config.MODEL.CONTOUR_DIM//2
        self.cloth_dim = config.MODEL.CLOTHES_DIM//2

        self.uncloth_net = GEN(in_feat_dim = self.uncloth_dim, out_img_dim=1, config = config)
        self.contour_net = GEN(in_feat_dim = self.contour_dim + self.cloth_dim, out_img_dim=1, config = config)
        self.cloth_net = GEN(in_feat_dim = self.cloth_dim, out_img_dim=1, config = config)


        
    def forward(self, x):
        x = self.base(x)
        x_ori = x
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        f_unclo = x_ori[:, 0:self.uncloth_dim, :, :]
        f_cont  = x_ori[:, self.uncloth_dim:self.uncloth_dim+self.contour_dim+self.cloth_dim, :, :]
        f_clo   = x_ori[:, self.uncloth_dim+self.contour_dim:self.uncloth_dim+self.contour_dim+self.cloth_dim, :, :]
        
        unclo_img = self.uncloth_net(f_unclo)
        cont_img  = self.contour_net(f_cont)
        clo_img   = self.cloth_net(f_clo)

        return (f, unclo_img, cont_img, clo_img)
