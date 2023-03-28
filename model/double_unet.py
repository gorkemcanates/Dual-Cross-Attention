import torch
import torch.nn as nn
from torchvision.models import vgg19
from model.utils.main_blocks import conv_block, DoubleASPP, params
from model.utils.main_blocks import SqueezeExciteBlock
from model.utils.dca import DCA


class local_conv_block(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.conv1 = conv_block(in_features=in_features, 
                                out_features=out_features)
        self.conv2 = conv_block(in_features=out_features, 
                                out_features=out_features, 
                                norm_type=None, 
                                activation=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.se = SqueezeExciteBlock(in_features=out_features, 
                                    reduction=8)
    def forward(self, x):
        x = self.conv1(x)
        xs = self.conv2(x)
        x = self.bn(xs)
        x = self.relu(x)
        x = self.se(x)
        return x, xs

class DoubleUnet(nn.Module):
    def __init__(self,
                attention=False, 
                n=1,
                in_features=3, 
                out_features=3, 
                k=1,
                input_size=(512, 512),
                patch_size=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],
                device='cuda', 
                ) -> None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)

        self.attention = attention    
        patch = input_size[0] // patch_size
  
        pretrained = False

        self.mu = torch.tensor([0.485, 0.456, 0.406],
                               requires_grad=False).to(device).view(
                                   (1, 3, 1, 1))
        self.sigma = torch.tensor([0.229, 0.224, 0.225],
                                  requires_grad=False).to(device).view(
                                      (1, 3, 1, 1))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), 
                                    stride=(2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2), 
                                    mode='bilinear', 
                                    align_corners=True)

        self.vgg1 = vgg19(pretrained=pretrained).features[:3]
        self.vgg2 = vgg19(pretrained=pretrained).features[4:8]
        self.vgg3 = vgg19(pretrained=pretrained).features[9:17]
        self.vgg4 = vgg19(pretrained=pretrained).features[18:26]
        self.vgg5 = vgg19(pretrained=pretrained).features[27:-2]

        for m in [self.vgg1, self.vgg2, self.vgg3, self.vgg4, self.vgg5]:
            for param in m.parameters():
                param.requires_grad = True

        self.aspp_1 = DoubleASPP(in_features=512, 
                                 out_features=64)
        if self.attention:
            self.DCA_vgg1 = DCA(n=n,                                            
                                    features = [64, 128, 256, 512],                                                                                                              
                                    strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                    patch=patch,
                                    spatial_att=spatial_att,
                                    channel_att=channel_att, 
                                    spatial_head=spatial_head_dim,
                                    channel_head=channel_head_dim,
                                    ) 
            self.DCA_vgg2 = DCA(n=n,                                            
                                    features = [64, 128, 256, 512],                                                                                                              
                                    strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8],
                                    patch_size=patch_size,
                                    spatial_att=spatial_att,
                                    channel_att=channel_att, 
                                    spatial_head=spatial_head_dim,
                                    channel_head=channel_head_dim,
                                    ) 
            self.DCA =     DCA(n=n,                                            
                                    features = [32, 64, 128, 256],                                                                                                              
                                    strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8],
                                    patch_size=patch_size,
                                    spatial_att=spatial_att,
                                    channel_att=channel_att, 
                                    spatial_head=spatial_head_dim,
                                    channel_head=channel_head_dim,
                                    ) 

        self.decode1 = local_conv_block(in_features=64 + 512, 
                                        out_features=256)
        self.decode2 = local_conv_block(in_features=256 + 256, 
                                        out_features=128)
        self.decode3 = local_conv_block(in_features=128 + 128, 
                                        out_features=64)
        self.decode4 = local_conv_block(in_features=64 + 64, 
                                         out_features=32)
        self.out1 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=in_features, 
                                            kernel_size=(1, 1), 
                                            padding=(0, 0)), 
                              nn.Sigmoid())

        self.encode2_1 = local_conv_block(in_features=in_features, 
                                            out_features=32)
        self.encode2_2 = local_conv_block(in_features=32, 
                                            out_features=64)
        self.encode2_3 = local_conv_block(in_features=64, 
                                            out_features=128)
        self.encode2_4 = local_conv_block(in_features=128, 
                                            out_features=256)

        self.aspp_2 = DoubleASPP(in_features=256, 
                                 out_features=64)

        self.decode2_1 = local_conv_block(in_features=64 + 512 + 256, 
                                        out_features=256)
        self.decode2_2 = local_conv_block(in_features=256 + 256 + 128, 
                                        out_features=128)
        self.decode2_3 = local_conv_block(in_features=128 + 128 + 64, 
                                        out_features=64)
        self.decode2_4 = local_conv_block(in_features=64 + 64 + 32, 
                                         out_features=32)
        self.out = nn.Conv2d(in_channels=32, 
                              out_channels=out_features, 
                              kernel_size=(1, 1), 
                              padding=(0, 0))
    def forward(self, x_in):
        if x_in.shape[1] == 1:
            x_in = torch.cat((x_in, x_in, x_in), dim=1)
        x_in = self.normalize(x_in)
        x1 = self.vgg1(x_in)
        x = self.relu(x1)
        x2 = self.vgg2(x)
        x = self.relu(x2)
        x3 = self.vgg3(x)
        x = self.relu(x3)
        x4 = self.vgg4(x)
        x = self.relu(x4)
        x = self.vgg5(x)
        x = self.relu(x)
        x = self.aspp_1(x)
        if self.attention:
            x1, x2, x3, x4 = self.DCA_vgg1([x1, x2, x3, x4])
            x12, x22, x32, x42 = self.DCA_vgg2([x1, x2, x3, x4])
        x = self.upsample(x)
        x = torch.cat((x4, x), dim=1)
        x, _ = self.decode1(x)
        x = self.upsample(x)
        x = torch.cat((x3, x), dim=1)
        x, _ = self.decode2(x)
        x = self.upsample(x)
        x = torch.cat((x2, x), dim=1)
        x, _ = self.decode3(x)
        x = self.upsample(x)
        x = torch.cat((x1, x), dim=1)
        x, _ = self.decode4(x)
        x = self.out1(x)
        out = x * x_in
        x, x1_2 = self.encode2_1(out)
        x = self.maxpool(x)
        x, x2_2 = self.encode2_2(x)
        x = self.maxpool(x)
        x, x3_2 = self.encode2_3(x)
        x = self.maxpool(x)
        x, x4_2 = self.encode2_4(x)
        x = self.maxpool(x)
        x = self.aspp_2(x)
        if self.attention:
            x1_2, x2_2, x3_2, x4_2 = self.DCA([x1_2, x2_2, x3_2, x4_2])
        x = self.upsample(x)
        x = torch.cat((x42, x4_2, x), dim=1)
        x, _ = self.decode2_1(x)
        x = self.upsample(x)
        x = torch.cat((x32, x3_2, x), dim=1)
        x, _ = self.decode2_2(x)
        x = self.upsample(x)
        x = torch.cat((x22, x2_2, x), dim=1)
        x, _ = self.decode2_3(x)
        x = self.upsample(x)
        x = torch.cat((x12, x1_2, x), dim=1)
        x, _ = self.decode2_4(x)
        x = self.out(x)            
        return x

    def normalize(self, x):
        return (x - self.mu) / self.sigma
