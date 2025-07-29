# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
    
#     def forward(self, x):
#         return x

# class Conv2dStaticSamePadding(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
#         if padding is None:
#             if isinstance(kernel_size, (tuple, list)):
#                 k = kernel_size[0]
#             else:
#                 k = kernel_size
#             if isinstance(dilation, (tuple, list)):
#                 d = dilation[0]
#             else:
#                 d = dilation
#             if isinstance(stride, (tuple, list)):
#                 s = stride[0]
#             else:
#                 s = stride
#             if s == 1:
#                 pad = (k - 1) * d // 2
#                 padding = (pad, pad, pad, pad)
#             elif s == 2 and k == 3:
#                 padding = (0, 1, 0, 1)
#             elif s == 2 and k == 5:
#                 padding = (2, 2, 2, 2)
#             else:
#                 padding = (0, 0, 0, 0)
#         else:
#             padding = padding
#         if dilation == 2:
#             super(Conv2dStaticSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding[:2], dilation=dilation, groups=groups, bias=bias)
#         else:
#             super(Conv2dStaticSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=(0,0), dilation=dilation, groups=groups, bias=bias)
#         self.static_padding = ZeroPad2d(padding) if padding != (0, 0, 0, 0) and dilation == 1 else Identity()

#     def forward(self, x):
#         x = self.static_padding(x)
#         x = super().forward(x) 
#         return x

# class ZeroPad2d(nn.Module):
#     def __init__(self, padding=(0,0,0,0)):
#         super(ZeroPad2d, self).__init__()
#         self.padding = padding

#     def forward(self, x):
#         return F.pad(x, self.padding)
    
#     def __repr__(self):
#         return f"ZeroPad2d({self.padding})"

# class MBConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, dilation=1):
#         super(MBConvBlock, self).__init__()
#         expanded_channels = in_channels * expand_ratio
#         stride_tuple = stride if isinstance(stride, (tuple, list)) else (stride, stride)
#         if expand_ratio != 1:
#             self._expand_conv = Conv2dStaticSamePadding(in_channels, expanded_channels, kernel_size=1, stride=1, bias=False, dilation=dilation)
#             self._bn0 = nn.BatchNorm2d(expanded_channels, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
#         else:
#             self._expand_conv = Identity()
#             self._bn0 = Identity()
        
#         self._depthwise_conv = Conv2dStaticSamePadding(
#             expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride_tuple, 
#             groups=expanded_channels, bias=False, dilation=dilation
#         )
#         self._bn1 = nn.BatchNorm2d(expanded_channels, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        
#         se_reduce_channels = in_channels // 4
#         self._se_reduce = Conv2dStaticSamePadding(expanded_channels, se_reduce_channels, kernel_size=1, stride=1, dilation=dilation)
#         self._se_expand = Conv2dStaticSamePadding(se_reduce_channels, expanded_channels, kernel_size=1, stride=1, dilation=dilation)
        
#         self._project_conv = Conv2dStaticSamePadding(expanded_channels, out_channels, kernel_size=1, stride=1, bias=False, dilation=dilation)
#         self._bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        
#         self._swish = nn.SiLU()
#         self.stride = stride
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x):
#         identity = x
#         if not isinstance(self._expand_conv, Identity):
#             x = self._expand_conv(x)
#             x = self._bn0(x)
#             x = self._swish(x)
        
#         x = self._depthwise_conv(x)
#         x = self._bn1(x)
#         x = self._swish(x)
        
#         se = F.avg_pool2d(x, kernel_size=x.size()[2:])
#         se = self._se_reduce(se)
#         se = self._swish(se)
#         se = self._se_expand(se)
#         se = torch.sigmoid(se)
#         x = x * se
        
#         x = self._project_conv(x)
#         x = self._bn2(x)
        
#         if self.stride == 1 and self.in_channels == self.out_channels:
#             x = x + identity
#         return x

# class EfficientNetEncoder(nn.Module):
#     def __init__(self):
#         super(EfficientNetEncoder, self).__init__()
#         self._conv_stem = Conv2dStaticSamePadding(1, 48, kernel_size=3, stride=2, bias=False)
#         self._bn0 = nn.BatchNorm2d(48, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
#         self._blocks = nn.ModuleList([
#             MBConvBlock(48, 24, kernel_size=3, stride=1, expand_ratio=1),
#             MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=1),
#             MBConvBlock(24, 32, kernel_size=3, stride=2, expand_ratio=6),
#             *[MBConvBlock(32, 32, kernel_size=3, stride=1, expand_ratio=6) for _ in range(3)],
#             MBConvBlock(32, 56, kernel_size=5, stride=2, expand_ratio=6),
#             *[MBConvBlock(56, 56, kernel_size=5, stride=1, expand_ratio=6) for _ in range(3)],
#             MBConvBlock(56, 112, kernel_size=3, stride=2, expand_ratio=6),
#             *[MBConvBlock(112, 112, kernel_size=3, stride=1, expand_ratio=6) for _ in range(5)],
#             MBConvBlock(112, 160, kernel_size=5, stride=1, expand_ratio=6),
#             *[MBConvBlock(160, 160, kernel_size=5, stride=1, expand_ratio=6) for _ in range(5)],
#             MBConvBlock(160, 272, kernel_size=5, stride=1, expand_ratio=6, dilation=2),
#             *[MBConvBlock(272, 272, kernel_size=5, stride=1, expand_ratio=6, dilation=2) for _ in range(7)],
#             MBConvBlock(272, 448, kernel_size=3, stride=1, expand_ratio=6, dilation=2),
#             MBConvBlock(448, 448, kernel_size=3, stride=1, expand_ratio=6, dilation=2),
#         ])
#         self._conv_head = Conv2dStaticSamePadding(448, 1792, kernel_size=1, stride=1, bias=False)
#         self._bn1 = nn.BatchNorm2d(1792, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
#         self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
#         self._dropout = nn.Dropout(p=0.4, inplace=False)
#         self._swish = nn.SiLU()

#     def forward(self, x):
#         x = self._conv_stem(x)
#         x = self._bn0(x)
#         for idx, block in enumerate(self._blocks):
#             x = block(x)
#             if idx == 5:
#                 low_level = x
#         high_level = x
#         x = self._conv_head(x)
#         x = self._bn1(x)
#         x = self._swish(x)
#         x = self._avg_pooling(x)
#         x = self._dropout(x)
#         return high_level, low_level

# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
#         super(SeparableConv2d, self).__init__()
#         if isinstance(padding, int):
#             padding_tuple = (padding, padding)
#         elif isinstance(padding, tuple) and len(padding) == 2:
#             padding_tuple = padding
#         else:
#             padding_tuple = (0, 0)
#         self._modules['0'] = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding_tuple, dilation=dilation, groups=in_channels, bias=bias)
#         self._modules['1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

#     def forward(self, x):
#         x = self._modules['0'](x)
#         x = self._modules['1'](x)
#         return x

# class ASPPSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation):
#         super(ASPPSeparableConv, self).__init__()
#         self._modules['0'] = SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
#         self._modules['1'] = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self._modules['2'] = nn.ReLU()

#     def forward(self, x):
#         x = self._modules['0'](x)
#         x = self._modules['1'](x)
#         x = self._modules['2'](x)
#         return x

# class ASPPPooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__()
#         self._modules['0'] = nn.AdaptiveAvgPool2d(output_size=1)
#         self._modules['1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
#         self._modules['2'] = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self._modules['3'] = nn.ReLU()
        

#     def forward(self, x):
#         size = x.shape[2:]
#         x = self._modules['0'](x)
#         x = self._modules['1'](x)
#         x = self._modules['2'](x)
#         x = self._modules['3'](x)
#         x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#         return x

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU()
#             ),
#             ASPPSeparableConv(in_channels, out_channels, dilation=12),
#             ASPPSeparableConv(in_channels, out_channels, dilation=24),
#             ASPPSeparableConv(in_channels, out_channels, dilation=36),
#             ASPPPooling(in_channels, out_channels)
#         ])
#         self.project = nn.Sequential(
#             nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Dropout(p=0.5, inplace=False)
#         )

#     def forward(self, x):
#         res = [conv(x) for conv in self.convs]
#         x = torch.cat(res, dim=1)
#         x = self.project(x)
#         return x

# class DeepLabV3PlusDecoder(nn.Module):
#     def __init__(self):
#         super(DeepLabV3PlusDecoder, self).__init__()
#         self.aspp = nn.Sequential(
#             ASPP(448, 256),
#             SeparableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU()
#         )
#         self.up = nn.UpsamplingBilinear2d(scale_factor=4.0)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(32, 48, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU()
#         )
#         self.block2 = nn.Sequential(
#             SeparableConv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU()
#         )

#     def forward(self, high_level, low_level):
#         x = self.aspp(high_level)
#         x = self.up(x)
#         low = self.block1(low_level)
#         x = torch.cat([x, low], dim=1)
#         x = self.block2(x)
#         return x

# class Activation(nn.Module):
#     def __init__(self):
#         super(Activation, self).__init__()
#         self.activation = Identity()

#     def forward(self, x):
#         return self.activation(x)

# class SegmentationHead(nn.Module):
#     def __init__(self, num_classes=6):
#         super(SegmentationHead, self).__init__()
#         self._modules['0'] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
#         self._modules['1'] = nn.UpsamplingBilinear2d(scale_factor=4.0)
#         self._modules['2'] = Activation()

#     def forward(self, x):
#         x = self._modules['0'](x)
#         x = self._modules['1'](x)
#         x = self._modules['2'](x)
#         return x

# class DeepLabV3Plus(nn.Module):
#     def __init__(self):
#         super(DeepLabV3Plus, self).__init__()
#         self.encoder = EfficientNetEncoder()
#         self.decoder = DeepLabV3PlusDecoder()
#         self.segmentation_head = SegmentationHead()

#     def forward(self, x):
#         high_level, low_level = self.encoder(x)
#         x = self.decoder(high_level, low_level)
#         x = self.segmentation_head(x)
#         return x


import segmentation_models_pytorch as smp
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b4',
            encoder_weights=None,
            activation=None,
            in_channels=1,
            classes=6
        )

    def forward(self, x):
        output = self.model(x)
        return output



