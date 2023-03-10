__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from model.utils.main_blocks import conv_block, double_conv_block_a, double_conv_block, Upconv, params
from model.utils.dca import DCA

class Unet(nn.Module):
    def __init__(self,
                attention=False, 
                n=1,
                in_features=3, 
                out_features=3, 
                k=0.5,
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
   
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.ReLU()
        norm2 = None
        self.conv1 = double_conv_block_a(in_features=in_features, 
                                        out_features1=int(64 * k), 
                                        out_features2=int(64 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * k))
        self.conv2 = double_conv_block_a(in_features=int(64 * k), 
                                        out_features1=int(128 * k), 
                                        out_features2=int(128 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * k))

        self.conv3 = double_conv_block_a(in_features=int(128 * k), 
                                        out_features1=int(256 * k), 
                                        out_features2=int(256 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * k))

        self.conv4 = double_conv_block_a(in_features=int(256 * k), 
                                        out_features1=int(512 * k), 
                                        out_features2=int(512 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)  
        self.norm4 = nn.BatchNorm2d(int(512 * k))
    
        self.conv5 = double_conv_block(in_features=int(512 * k), 
                                        out_features1=int(1024 * k), 
                                        out_features2=int(1024 * k), 
                                        norm_type='bn')   


        if self.attention:
            self.DCA = DCA(n=n,                                            
                                features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                                            )  
          

        self.up1 = Upconv(in_features=int(1024 * k), 
                            out_features=int(512 * k), 
                            norm_type='bn')

        self.upconv1 = double_conv_block(in_features=int(512 * k + 512 * k), 
                                        out_features1=int(512 * k), 
                                        out_features2=int(512 * k), 
                                        norm_type='bn')

        self.up2 = Upconv(in_features=int(512 * k), 
                            out_features=int(256 * k), 
                            norm_type='bn')


        self.upconv2 = double_conv_block(in_features=int(256 * k + 256 * k), 
                                        out_features1=int(256 * k), 
                                        out_features2=int(256 * k), 
                                        norm_type='bn')

        self.up3 = Upconv(in_features=int(256 * k), 
                            out_features=int(128 * k), 
                            norm_type='bn')


        self.upconv3 = double_conv_block(in_features=int(128 * k + 128 * k), 
                                        out_features1=int(128 * k), 
                                        out_features2=int(128 * k), 
                                        norm_type='bn')

        self.up4 = Upconv(in_features=int(128 * k), 
                            out_features=int(64 * k), 
                            norm_type='bn')

        self.upconv4 = double_conv_block(in_features=int(64 * k + 64 * k), 
                                        out_features1=int(64 * k), 
                                        out_features2=int(64 * k), 
                                        norm_type='bn')    

        self.out = conv_block(in_features=int(64 * k), 
                            out_features=out_features, 
                            norm_type=None,
                            activation=False, 
                            kernel_size=(1, 1), 
                            padding=(0, 0))   

        # self.initialize_weights()                                     

    def forward(self, x):
        x1 = self.conv1(x)
        x1_n = self.norm1(x1)
        x1_a = self.relu(x1_n)
        x2 = self.maxpool(x1_a)
        x2 = self.conv2(x2)
        x2_n = self.norm2(x2)
        x2_a = self.relu(x2_n)
        x3 = self.maxpool(x2_a) 
        x3 = self.conv3(x3)
        x3_n = self.norm3(x3)
        x3_a = self.relu(x3_n)
        x4 = self.maxpool(x3_a)
        x4 = self.conv4(x4)
        x4_n = self.norm4(x4)
        x4_a = self.relu(x4_n)
        x5 = self.maxpool(x4_a)
        x = self.conv5(x5)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1, x2, x3, x4])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconv4(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    model = Unet()
    in1 = torch.rand(1, 3, 512, 512)
    out = model(in1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(out.shape)
