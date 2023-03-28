# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import torch
import torch.nn as nn
import einops
import matplotlib.pyplot as plt
from model.utils.main_blocks import *

def params(module):
    return sum(p.numel() for p in module.parameters())


class UpsampleConv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                padding=(1, 1), 
                norm_type=None, 
                activation=False,
                scale=(2, 2), 
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='bilinear', 
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    norm_type=norm_type, 
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type, 
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class depthwise_projection(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1), 
                padding=(0, 0), 
                norm_type=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_type=norm_type,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x


class conv_projection(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.proj = conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=(1, 1), 
                                        padding=(0, 0),
                                        norm_type=None,
                                        activation=False)
    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')        
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                size,
                patch=28,
                proj='conv'
                ) -> None:
        super().__init__()
        self.proj = proj
        if self.proj == 'conv':
            self.projection = nn.Conv2d(in_channels=in_features, 
                                        out_channels=out_features, 
                                        kernel_size=size // patch_size, 
                                        stride=size // patch_size, 
                                        padding=(0, 0), 
                                        )
        
    def forward(self, x):
        x = self.projection(x) 
        x = x.flatten(2).transpose(1, 2)      
        return x


class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')        
        return x


class Layernorm(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps)                                                  
    def forward(self, x):
        H = x.shape[2]
        x = einops.rearrange(x, 'B C H W -> B (H W) C')        
        x = self.norm(x)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=H) 
        return x       


class ScaleDotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
                                                    
    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3) 
        return x123
