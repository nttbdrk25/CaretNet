import re
import types

import torch.nn
import torch.nn.init
import torchvision.transforms as Tr
from .common import conv1x1_block, conv3x3_block, Classifier,conv3x3_dw_blockAll
from .SE_Attention import *
class BIFblock(torch.nn.Module):#interleave tensors
    """
    interleave block internally used in CaretNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,init_status):
        super().__init__()
        self.pw_in = conv1x1_block(in_channels=in_channels,
                                out_channels=in_channels, use_bn=False, activation=None)
        self.pw_grouped = conv1x1_block(in_channels=2*in_channels,
                                out_channels=in_channels, use_bn=False,
                                activation=None, groups=in_channels)
        self.dw_3x3 = conv3x3_dw_blockAll(channels=in_channels, stride=stride)
        self.dw_dilated_3x3 = conv3x3_dw_blockAll(channels=in_channels, stride=stride,padding=2,dilation=2)        
        self.pw_out = conv1x1_block(in_channels=in_channels, out_channels=out_channels)        
        if init_status == True or in_channels < out_channels:
            self.pw_residual = conv1x1_block(in_channels=in_channels,
                                    out_channels=out_channels, stride=stride)           
        self.SE = SE(out_channels, 16)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_status = init_status
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        residual = x        
        x = self.pw_in(x)
        x_dw = self.dw_3x3(x)
        x_dilated_dw = self.dw_dilated_3x3(x)
        x = self.interleave(x_dw,x_dilated_dw)        
        x = self.pw_grouped(x)
        x = self.pw_out(x)        
        x = self.SE(x)          
        if self.init_status == False and self.in_channels > self.out_channels:
            n,c,h,w = residual.size()
            k=2
            residual = residual.reshape(n, c//k, k, h, w).mean(2)            
            if self.stride == 2:
                residual = self.maxpool(residual)                
            x = x + residual
        else:
            residual = self.pw_residual(residual)            
            x = x + residual
        return x
    def interleave(self,x1,x2):##x1 has the same dimesion as x2
        batch_size, channels, height, width = x1.size()
        x = torch.stack((x1,x2), dim=2).view(batch_size,2*channels,height,width)
        return x
class CaretNet(torch.nn.Module):
    """
    Class for constructing CaretNet.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        init_status = True
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), BIFblock(in_channels=in_channels, out_channels=unit_channels, stride=stride,init_status=init_status))
                init_status = False
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 512
        self.backbone.add_module("final_conv1", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        in_channels = self.final_conv_channels
        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv2", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))        
        self.backbone.add_module("dropout1",torch.nn.Dropout2d(0.2))#with dropout
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.backbone.add_module("dropout2",torch.nn.Dropout2d(0.2))#with dropout
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)            
            
            elif isinstance(module, torch.nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            
        # classifier
        self.classifier.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
def build_CaretNet(num_classes, typesize=1.0, cifar=False):
    """
    Construct a CaretNet from the given set of parameters.   
    Args:
        num_classes (int): Number of classes for the classification layer.        
        cifar (bool): if `True`, make the model suitable for the CIFAR10/100
            datasets. Otherwise, the model will be suited for ImageNet and
            fine-grained datasets.
        
    Returns:
        The constructed CaretNet.
    """
    init_conv_channels = 32         
    channels = [[64], [128, 64, 128], [256,128,256],[512],[256]]    
    if typesize!=1.0:        
        channels = [[int(unit * typesize) for unit in stage] for stage in channels]        
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]        
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    return  CaretNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)