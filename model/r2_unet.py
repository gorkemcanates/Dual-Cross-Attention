import torch
import torch.nn as nn
from model.utils.main_blocks import rrcnn_block, Upconv, params
from model.utils.dca import DCA

class R2Unet(nn.Module):
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
        
        self.rconv1 = rrcnn_block(in_features=in_features, 
                                  out_features=int(64 * k))
        self.rconv2 = rrcnn_block(in_features=int(64 * k), 
                                  out_features=int(128 * k))
        self.rconv3 = rrcnn_block(in_features=int(128 * k), 
                                  out_features=int(256 * k))
        self.rconv4 = rrcnn_block(in_features=int(256 * k), 
                                  out_features=int(512 * k))
        self.rconv5 = rrcnn_block(in_features=int(512 * k), 
                                  out_features=int(1024 * k))

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
                            )          
        self.rconv6 = rrcnn_block(in_features=int(1024 * k), 
                                  out_features=int(512 * k))

        self.up2 = Upconv(in_features=int(512 * k), 
                            out_features=int(256 * k), 
                            )          
        self.rconv7 = rrcnn_block(in_features=int(512 * k), 
                                  out_features=int(256 * k))

        self.up3 = Upconv(in_features=int(256 * k), 
                            out_features=int(128 * k), 
                            )          
        self.rconv8 = rrcnn_block(in_features=int(256 * k), 
                                  out_features=int(128 * k))

        self.up4 = Upconv(in_features=int(128 * k), 
                            out_features=int(64 * k), 
                            )          
        self.rconv9 = rrcnn_block(in_features=int(128 * k), 
                                  out_features=int(64 * k))   
        self.out = nn.Conv2d(in_channels=int(64 * k),
                             out_channels=out_features,
                             kernel_size=(1, 1), 
                             padding=(0, 0))                                                          


    def forward(self, x):
        x1, x1_ = self.rconv1(x)
        x2 = self.maxpool(x1)
        x2, x2_ = self.rconv2(x2)
        x3 = self.maxpool(x2)
        x3, x3_ = self.rconv3(x3)
        x4 = self.maxpool(x3)
        x4, x4_ = self.rconv4(x4)
        x = self.maxpool(x4)
        x, _ = self.rconv5(x)
        if self.attention:
            x1, x2, x3, x4 = self.DCA([x1_, x2_, x3_, x4_])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x, _ = self.rconv6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x, _ = self.rconv7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x, _ = self.rconv8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x, _ = self.rconv9(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = R2Unet()
    in1 = torch.rand(1, 3, 224, 224)
    out = model(in1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(out.shape)

