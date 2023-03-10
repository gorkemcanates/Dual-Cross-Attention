__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from model.utils.dca import DCA

class local_conv_block(nn.Module):
    def __init__(self, in_features, out_features, norm=True, activation=True) -> None:
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_features, 
                                            out_channels=out_features, 
                                            kernel_size=(5, 5), 
                                            padding=(2, 2))
        if self.norm:
            self.bn = nn.BatchNorm2d(out_features)
        if self.activation:
            self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.prelu(x)
        return x 

class InputConv(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, 
                              out_channels=out_features, 
                              kernel_size=(5, 5), 
                              padding=(2, 2))
        self.conv_skip = nn.Conv2d(in_channels=in_features, 
                                   out_channels=out_features,
                                  kernel_size=(1, 1), 
                                  padding=(0, 0))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_skip = self.conv_skip(x)
        x = self.conv(x)
        xs = x + x_skip
        x = self.bn(x)
        x = self.prelu(x)
        return x, xs

class DownConv(nn.Module):
    def __init__(self, in_features, out_features, n) -> None:
        super().__init__()
        self.n = n
        self.down = nn.Conv2d(in_channels=in_features, 
                              out_channels=out_features, 
                              kernel_size=(2, 2), 
                              stride=(2, 2)
                              )

        self.conv = nn.ModuleList([local_conv_block(in_features=out_features, 
                                            out_features=out_features)
                                            for _ in range(n - 1)])

        self.conv.append(local_conv_block(in_features=out_features, 
                                          out_features=out_features, 
                                          norm=False, 
                                          activation=False))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_d = self.down(x)
        x = x_d.clone()
        for i in range(self.n):
            x = self.conv[i](x)
        x_s = x + x_d     
        x = self.bn(x_s)
        x = self.prelu(x)
        return x, x_s 

class UpConv(nn.Module):
    def __init__(self, in_features, enc_features, out_features, n) -> None:
        super().__init__()
        self.n = n
        self.up = nn.ConvTranspose2d(in_channels=in_features, 
                                    out_channels=out_features, 
                                    kernel_size=(2, 2), 
                                    stride=(2, 2)
                              )
        self.bn_in = nn.BatchNorm2d(out_features)
        self.prelu_in = nn.PReLU()
        self.conv = nn.ModuleList([local_conv_block(in_features=out_features + enc_features, 
                                                    out_features=out_features)])

        for _ in range(n - 2):
            self.conv.append(local_conv_block(in_features=out_features, 
                                              out_features=out_features))

        self.conv.append(local_conv_block(in_features=out_features, 
                                          out_features=out_features, 
                                          norm=False, 
                                          activation=False))
        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.bn_in(x_d)
        x = self.prelu_in(x)
        x = torch.cat((x_e, x), dim=1)
        for i in range(self.n):
            x = self.conv[i](x)
        x += x_d
        x = self.bn(x)
        x = self.prelu(x)
        return x    

class OutputConv(nn.Module):
    def __init__(self, in_features, enc_features, out_features) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_features, 
                                    out_channels=out_features, 
                                    kernel_size=(2, 2), 
                                    stride=(2, 2)
                              )
        self.bn_in = nn.BatchNorm2d(out_features)
        self.prelu_in = nn.PReLU()                              
        self.conv = nn.Conv2d(in_channels=out_features + enc_features, 
                              out_channels=out_features, 
                              kernel_size=(5, 5), 
                              padding=(2, 2))

        self.bn = nn.BatchNorm2d(out_features)
        self.prelu = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.bn_in(x_d)
        x = self.prelu_in(x)
        x = torch.cat((x_e, x), dim=1)
        x = self.conv(x)
        x += x_d
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Vnet(nn.Module):
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
        k = 1
        if device == 'cuda':
            torch.cuda.set_enabled_lms(True)

        self.attention = attention    
        patch = input_size[0] // patch_size
  
        
        self.conv1 = InputConv(in_features=in_features, 
                               out_features=int(32 * k))
        self.conv2 = DownConv(in_features=int(32 * k), 
                              out_features=int(64 * k), 
                              n=2)
        self.conv3 = DownConv(in_features=int(64 * k), 
                              out_features=int(128 * k), 
                              n=3)
        self.conv4 = DownConv(in_features=int(128 * k), 
                              out_features=int(256 * k), 
                              n=3)
        self.conv5 = DownConv(in_features=int(256 * k), 
                              out_features=int(512 * k), 
                              n=3)
        
        if self.attention:
            self.DCA = DCA(n=n,                                            
                                features = [int(32 * k), int(64 * k), int(128 * k), int(256 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                )  
        self.up1 = UpConv(in_features=int(512 * k),
                          enc_features=int(256 * k), 
                          out_features=int(256 * k), 
                          n=3)
        self.up2 = UpConv(in_features=int(256 * k),
                          enc_features=int(128 * k), 
                          out_features=int(128 * k), 
                          n=3)
        self.up3 = UpConv(in_features=int(128 * k),
                          enc_features=int(64 * k), 
                          out_features=int(64 * k), 
                          n=2)
        self.up4 = OutputConv(in_features=int(64 * k),
                              enc_features=int(32 * k), 
                              out_features=int(32 * k))

        self.out = nn.Conv2d(in_channels=int(32 * k), 
                              out_channels=out_features, 
                              kernel_size=(1, 1), 
                              padding=(0, 0))
    def forward(self, x):
        x1, x1_ = self.conv1(x)
        x2, x2_ = self.conv2(x1)
        x3, x3_ = self.conv3(x2)
        x4, x4_ = self.conv4(x3)
        x, _ = self.conv5(x4)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = Vnet()
    in1 = torch.rand(1, 3, 512, 512)
    out = model(in1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(out.shape)
