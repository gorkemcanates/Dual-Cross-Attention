import torch
import torch.nn as nn
from model.utils.main_blocks import Upconv, double_conv_block, double_conv_block_a, conv_block
from model.utils.attention_blocks import AttentionGate
from model.utils.fusion_blocks import Fusion

class AttUnet(nn.Module):
    def __init__(self,
                enc_fusion=False, 
                in_features=3, 
                out_features=3, 
                k=0.5,
                input_size=(512, 512),
                fusion_out=None,
                patch_size_ratio=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=8,
                channel_head_dim=8, 
                device='cuda', 
                ) -> None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)
        self.enc_fusion = enc_fusion    
        patch_size = input_size[0] // patch_size_ratio
        if fusion_out is not None:
                fusion_out = int(fusion_out * k)  

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
        if self.enc_fusion:
            self.fusion = Fusion(features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],                                                                                                              
                                            strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4, patch_size_ratio // 8],
                                            patch_size=patch_size,
                                            fusion_out=fusion_out,
                                            spatial_att=spatial_att,
                                            channel_att=channel_att, 
                                            spatial_head=spatial_head_dim,
                                            channel_head=channel_head_dim,
                                                            ) 

        self.up1 = Upconv(in_features=int(1024 * k), 
                            out_features=int(512 * k), 
                            norm_type='bn')

        
        self.att1 = AttentionGate(F_g=int(512 * k),
                                    F_l=int(1024 * k),
                                    F_int=int(256 * k), 
                                    )

        self.upconv1 = double_conv_block(in_features=int(512 * k + 512 * k), 
                                        out_features1=int(512 * k), 
                                        out_features2=int(512 * k), 
                                        norm_type='bn')

        self.up2 = Upconv(in_features=int(512 * k), 
                            out_features=int(256 * k), 
                            norm_type='bn')

        self.att2 = AttentionGate(F_g=int(256 * k), 
                                    F_l=int(512 * k),
                                    F_int=int(128 * k), 
                                    )

        self.upconv2 = double_conv_block(in_features=int(256 * k + 256 * k), 
                                        out_features1=int(256 * k), 
                                        out_features2=int(256 * k), 
                                        norm_type='bn')

        self.up3 = Upconv(in_features=int(256 * k), 
                            out_features=int(128 * k), 
                            norm_type='bn')

        self.att3 = AttentionGate(F_g=int(128 * k),
                                    F_l=int(256 * k),
                                    F_int=int(64 * k), 
                                    )
        
        self.upconv3 = double_conv_block(in_features=int(128 * k + 128 * k), 
                                        out_features1=int(128 * k), 
                                        out_features2=int(128 * k), 
                                        norm_type='bn')

        self.up4 = Upconv(in_features=int(128 * k), 
                            out_features=int(64 * k), 
                            norm_type='bn')

        self.att4 = AttentionGate(F_g=int(64 * k),
                                    F_l=int(128 * k),
                                    F_int=int(32 * k), 
                                    )

        self.upconv4 = double_conv_block(in_features=int(64 * k + 64 * k), 
                                        out_features1=int(64 * k), 
                                        out_features2=int(64 * k), 
                                        norm_type='bn')    

        self.out = conv_block(in_features=int(64 * k), 
                            out_features=out_features, 
                            kernel_size=(1, 1), 
                            padding=(0, 0),                            
                            norm_type=None,
                            activation=False
                            )  
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        x_1 = self.conv5(x5)
        if self.enc_fusion:
            x1, x2, x3, x4 = self.fusion(x1, x2, x3, x4)        
        x_1a = self.att1(x4, x_1)
        x = self.up1(x_1)
        x = torch.cat((x, x_1a), dim=1)
        x_2 = self.upconv1(x)
        x_2a = self.att2(x3, x_2)
        x = self.up2(x_2)
        x = torch.cat((x, x_2a), dim=1)
        x_3 = self.upconv2(x)
        x_3a = self.att3(x2, x_3)
        x = self.up3(x_3)
        x = torch.cat((x, x_3a), dim=1)
        x_4 = self.upconv3(x)
        x_4a = self.att4(x1, x_4)
        x = self.up4(x_4)
        x = torch.cat((x, x_4a), dim=1)
        x = self.upconv4(x)
        x = self.out(x)
        return x

