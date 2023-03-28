# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


import torch
import torch.nn as nn
import einops
from model.utils.main_blocks import *
from model.utils.dca_utils import *


class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features) 

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)
        self.sdp = ScaleDotProduct()        
        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5                     
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k ,v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att

class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)       

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)                                             
        self.sdp = ScaleDotProduct()        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)  
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5        
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k ,v, scale).transpose(1, 2).flatten(2)    
        x = self.projection(att)
        return x

class CCSABlock(nn.Module):
    def __init__(self, 
                features, 
                channel_head, 
                spatial_head, 
                spatial_att=True, 
                channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])   

            self.c_attention = nn.ModuleList([ChannelAttention(
                                                in_features=sum(features),
                                                out_features=feature,
                                                n_heads=head, 
                                        ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])          
          
            self.s_attention = nn.ModuleList([SpatialAttention(
                                                    in_features=sum(features),
                                                    out_features=feature,
                                                    n_heads=head, 
                                                    ) 
                                                    for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)   
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)   
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att    

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]        
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att 
        

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]    

    def cat(self, *args):
        return torch.cat((args), dim=2)



class DCA(nn.Module):
    def __init__(self,
                features,
                strides,
                patch=28,
                channel_att=True,
                spatial_att=True,   
                n=1,              
                channel_head=[1, 1, 1, 1], 
                spatial_head=[4, 4, 4, 4], 
                ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
                                                    pooling = nn.AdaptiveAvgPool2d,            
                                                    patch=patch, 
                                                    )
                                                    for _ in features])                
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                            out_features=feature, 
                                                            kernel_size=(1, 1),
                                                            padding=(0, 0), 
                                                            groups=feature
                                                            )
                                                    for feature in features])         
                                
        self.attention = nn.ModuleList([
                                        CCSABlock(features=features, 
                                                  channel_head=channel_head, 
                                                  spatial_head=spatial_head, 
                                                  channel_att=channel_att, 
                                                  spatial_att=spatial_att) 
                                                  for _ in range(n)])
                     
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, 
                                                    out_features=feature,
                                                    kernel_size=(1, 1),
                                                    padding=(0, 0),
                                                    norm_type=None,
                                                    activation=False,
                                                    scale=stride, 
                                                    conv='conv')
                                                    for feature, stride in zip(features, strides)])                                                      
        self.bn_relu = nn.ModuleList([nn.Sequential(
                                                    nn.BatchNorm2d(feature), 
                                                    nn.ReLU()
                                                    ) 
                                                    for feature in features])
    
    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg) 
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out, )      

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]  
        
    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch) 



