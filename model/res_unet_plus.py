__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from model.utils.main_blocks import conv_block, ResConv, ASPP
from model.utils.main_blocks import SqueezeExciteBlock, AttentionBlock
from model.utils.dca import DCA

class ResUnetPlus(nn.Module):
    def __init__(self,
                attention=False, 
                n=1,
                in_features=3, 
                out_features=3, 
                k=0.5,
                input_size=(512, 512),
                fusion_out=None,
                patch_size=4,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4],
                channel_head_dim=[1, 1, 1], 
                device='cuda', 
                ) -> None:
        super().__init__()
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)

        self.attention = attention    
        patch = input_size[0] // patch_size

        self.input_layer = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=int(64 * k),
                       ),
            nn.Conv2d(int(64 * k), int(64 * k), kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_features, int(64 * k), kernel_size=3, padding=1))

        self.squeeze_excite1 = SqueezeExciteBlock(int(64 * k),
                                                  reduction=int(16 * k),
                                                  )

        self.residual_conv1 = ResConv(int(64 * k),
                                      int(128 * k),
                                      stride=2,
                                      )

        self.squeeze_excite2 = SqueezeExciteBlock(int(128 * k),
                                                  reduction=int(32 * k),
                                                  )

        self.residual_conv2 = ResConv(int(128 * k),
                                      int(256 * k),
                                      stride=2,
                                      )

        self.squeeze_excite3 = SqueezeExciteBlock(int(256 * k),
                                                  reduction=int(32 * k),
                                                  )

        self.residual_conv3 = ResConv(int(256 * k),
                                      int(512 * k),
                                      stride=2,
                                      )

        self.aspp_bridge = ASPP(int(512 * k),
                                    int(1024 * k),
                                    norm_type='bn',
                                    activation=False
                                    )
        if self.attention:
            self.DCA = DCA(n=n,                                            
                                features = [int(64 * k), int(128 * k), int(256 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                ) 
        self.attn1 = AttentionBlock(int(256 * k),
                                    int(1024 * k),
                                    int(1024 * k),
                                    )
        self.upsample1 = nn.Upsample(scale_factor=2,
                                     mode='nearest',
                                     )

        self.up_residual_conv1 = ResConv(int(1024 * k) + int(256 * k),
                                         int(512 * k),
                                         )

        self.attn2 = AttentionBlock(int(128 * k),
                                    int(512 * k),
                                    int(512 * k),
                                    )
        self.upsample2 = nn.Upsample(scale_factor=2,
                                     mode='nearest',
                                     )

        self.up_residual_conv2 = ResConv(int(512 * k) + int(128 * k),
                                         int(256 * k),
                                         )

        self.attn3 = AttentionBlock(int(64 * k),
                                    int(256 * k),
                                    int(256 * k),
                                    )
        self.upsample3 = nn.Upsample(scale_factor=2,
                                     mode='nearest',
                                     )

        self.up_residual_conv3 = ResConv(int(256 * k) + int(64 * k),
                                         int(128 * k),
                                         )

        self.aspp_out = ASPP(int(128 * k), int(64 * k))

        self.output_layer = nn.Conv2d(int(64 * k),
                                      out_features,
                                      kernel_size=(1, 1), 
                                      padding=(0, 0))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        if self.attention:
            x1, x2, x3 = self.DCA([x1, x2, x3])       

        x6 = self.attn1(x3, x5)

        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out



if __name__ == '__main__':

    def test(batchsize):
        device = 'cpu'
        in_channels = 3
        in1 = torch.rand(batchsize, in_channels, 512, 512).to(device=device)
        model = ResUnetPlus(in_features=in_channels,
                            out_features=3,
                            k=1,
                            ).to(device=device)

        out1 = model(in1)
        total_params = sum(p.numel() for p in model.parameters())

        return out1.shape, total_params

    shape, total_params = test(batchsize=4)
    print('Shape : ', shape, '\nTotal params : ', "{:,}".format(total_params))
