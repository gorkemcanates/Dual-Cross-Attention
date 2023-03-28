import torch
import torch.nn as nn
from model.utils.main_blocks import conv_block, params
from model.utils.dca import DCA

class MultiResBlock(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()
         f1 = int(1.67 * filters * 0.167)
         f2 = int(1.67 * filters * 0.333)
         f3 = int(1.67 * filters * 0.5)
         fout = f1 + f2 + f3

         self.skip = conv_block(in_features=in_features,
                                out_features=fout, 
                                kernel_size=(1, 1), 
                                padding=(0, 0), 
                                norm_type='bn', 
                                activation=False)
         self.c1 = conv_block(in_features=in_features,
                                out_features=f1, 
                                kernel_size=(3, 3), 
                                padding=(1, 1), 
                                norm_type='bn', 
                                activation=True)
         self.c2 = conv_block(in_features=f1,
                                out_features=f2, 
                                kernel_size=(3, 3), 
                                padding=(1, 1), 
                                norm_type='bn', 
                                activation=True)
         self.c3 = conv_block(in_features=f2,
                                out_features=f3, 
                                kernel_size=(3, 3), 
                                padding=(1, 1), 
                                norm_type='bn', 
                                activation=True)
         self.bn1 = nn.BatchNorm2d(fout)
         self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn1(x)
        x += x_skip
        # x = self.bn2(x)
        x = self.relu(x)
        return x


class ResPath(nn.Module):
    def __init__(self, in_features, out_features, n) -> None:
         super().__init__()
         self.n = n

         self.bns = nn.ModuleList([nn.BatchNorm2d(out_features) for _ in range(n)])

         self.skips = nn.ModuleList([conv_block(in_features=in_features, 
                                                out_features=out_features, 
                                                kernel_size=(1, 1), 
                                                padding=(0, 0), 
                                                norm_type=None, 
                                                activation=False)])   

         self.convs = nn.ModuleList([conv_block(in_features=in_features, 
                                                out_features=out_features, 
                                                kernel_size=(3, 3), 
                                                padding=(1, 1), 
                                                norm_type=None, 
                                                activation=False), 
                                                ])

         for _ in range(n - 1):
            self.skips.append(conv_block(in_features=out_features, 
                                                    out_features=out_features, 
                                                    kernel_size=(1, 1), 
                                                    padding=(0, 0), 
                                                    norm_type=None, 
                                                    activation=False))                                                
            self.convs.append(conv_block(in_features=out_features, 
                                                    out_features=out_features, 
                                                    kernel_size=(3, 3), 
                                                    padding=(1, 1), 
                                                    norm_type=None, 
                                                    activation=False))    
         self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n):
            x_skip = self.skips[i](x)
            x = self.convs[i](x)
            x_s = x + x_skip
            x = self.bns[i](x_s)
            x = self.relu(x)
        return x, x_s

class MultiResUnet(nn.Module):
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

        alpha = 1.67
        k = 1
        in_filters1 = int(32*alpha*0.167)+int(32*alpha*0.333)+int(32*alpha* 0.5)                                
        in_filters2 = int(32*2*alpha*0.167)+int(32*2*alpha*0.333)+int(32*2*alpha* 0.5)
        in_filters3 = int(32*4*alpha*0.167)+int(32*4*alpha*0.333)+int(32*4*alpha* 0.5)
        in_filters4 = int(32*8*alpha*0.167)+int(32*8*alpha*0.333)+int(32*8*alpha* 0.5)
        in_filters5 = int(32*16*alpha*0.167)+int(32*16*alpha*0.333)+int(32*16*alpha* 0.5)        
        in_filters6 = int(32*8*alpha*0.167)+int(32*8*alpha*0.333)+int(32*8*alpha* 0.5)        
        in_filters7 = int(32*4*alpha*0.167)+int(32*4*alpha*0.333)+int(32*4*alpha* 0.5)        
        in_filters8 = int(32*2*alpha*0.167)+int(32*2*alpha*0.333)+int(32*2*alpha* 0.5)        
        in_filters9 = int(32*alpha*0.167)+int(32*alpha*0.333)+int(32*alpha* 0.5)        

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.mr1 = MultiResBlock(in_features=in_features, 
                                filters=int(32 * k))
        self.respath1 = ResPath(in_features=in_filters1, 
                                out_features=32, 
                                n=4)
        self.mr2 = MultiResBlock(in_features=in_filters1, 
                                filters=int(32 * k * 2))
        self.respath2 = ResPath(in_features=in_filters2, 
                                out_features=32 * 2, 
                                n=3)
        self.mr3 = MultiResBlock(in_features=in_filters2, 
                                filters=int(32 * k * 4))
        self.respath3 = ResPath(in_features=in_filters3, 
                                out_features=32 * 4, 
                                n=2)
        self.mr4 = MultiResBlock(in_features=in_filters3, 
                                filters=int(32 * k * 8))
        self.respath4 = ResPath(in_features=in_filters4, 
                                out_features=32 * 8, 
                                n=1)
        self.mr5 = MultiResBlock(in_features=in_filters4, 
                                filters=int(32 * k * 16))

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

        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_filters5, 
                                      32*8, 
                                    kernel_size=(2,2), 
                                    stride=(2,2)), 
                                    nn.BatchNorm2d(32 * 8), 
                                    nn.ReLU())


        self.mr6 = MultiResBlock(in_features=32 * 8 * 2, 
                                filters=int(32 * k * 8))  

        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_filters6, 
                                      32*4, 
                                    kernel_size=(2,2), 
                                    stride=(2,2)), 
                                    nn.BatchNorm2d(32 * 4), 
                                    nn.ReLU())     

        self.mr7 = MultiResBlock(in_features=32 * 4 * 2, 
                                filters=int(32 * k * 4)) 

        self.up3 = nn.Sequential(nn.ConvTranspose2d(in_filters7, 
                                      32*2, 
                                    kernel_size=(2,2), 
                                    stride=(2,2)), 
                                    nn.BatchNorm2d(32 * 2), 
                                    nn.ReLU())    
                                                      
        self.mr8 = MultiResBlock(in_features=32 * 2 * 2, 
                                filters=int(32 * k * 2))                                                        
        self.up4 = nn.Sequential(nn.ConvTranspose2d(in_filters8, 
                                      32, 
                                    kernel_size=(2,2), 
                                    stride=(2,2)), 
                                    nn.BatchNorm2d(32), 
                                    nn.ReLU())   
  
        self.mr9 = MultiResBlock(in_features=32 * 2, 
                                filters=int(32 * k))    

        self.out = nn.Conv2d(in_channels=in_filters9,
                             out_channels=out_features,
                             kernel_size=(1, 1), 
                             padding=(0, 0)
                            )                  

    def forward(self, x):
        x1 = self.mr1(x)
        xp1 = self.maxpool(x1)
        x1, x1_ = self.respath1(x1)     
        x2 = self.mr2(xp1)
        xp2 = self.maxpool(x2)
        x2, x2_ = self.respath2(x2)   

        x3 = self.mr3(xp2)
        xp3 = self.maxpool(x3)
        x3, x3_ = self.respath3(x3)   

        x4 = self.mr4(xp3)
        xp4 = self.maxpool(x4)
        x4, x4_ = self.respath4(x4)  

        x = self.mr5(xp4)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])

        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.mr6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.mr7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.mr8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.mr9(x)
        x = self.out(x)
        return x

        



