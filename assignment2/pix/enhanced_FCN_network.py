import torch
import torch.nn as nn

class EnhancedFullyConvNetwork(nn.Module):
    """
    增强版全卷积网络，带有跳跃连接，适用于Pix2Pix任务
    这种架构类似于U-Net，具有编码器-解码器结构和跳跃连接
    """
    def __init__(self):
        super(EnhancedFullyConvNetwork, self).__init__()
        
        # 编码器部分 (下采样)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3x256x256 -> 64x128x128
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x128x128 -> 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x64x64 -> 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 256x32x32 -> 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x16x16 -> 512x8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x8x8 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x4x4 -> 512x2x2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x2x2 -> 512x1x1
            nn.ReLU(inplace=True)  # 移除BatchNorm，因为在1x1特征图上无法使用
        )
        
        # 解码器部分 (上采样)，带跳跃连接
        # Pix2Pix论文中Dropout仅用于前3个解码层，且放在ReLU之后
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 512x1x1 -> 512x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1024x2x2 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1024x4x4 -> 512x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1024x8x8 -> 512x16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 1024x16x16 -> 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 512x32x32 -> 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder7 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 256x64x64 -> 64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # 128x128x128 -> 3x256x256
            nn.Tanh()
        )
        
    def forward(self, x):
        # 编码器前向传播
        enc1 = self.encoder1(x)      # 64x128x128
        enc2 = self.encoder2(enc1)   # 128x64x64
        enc3 = self.encoder3(enc2)   # 256x32x32
        enc4 = self.encoder4(enc3)   # 512x16x16
        enc5 = self.encoder5(enc4)   # 512x8x8
        enc6 = self.encoder6(enc5)   # 512x4x4
        enc7 = self.encoder7(enc6)   # 512x2x2
        enc8 = self.encoder8(enc7)   # 512x1x1 (移除了BatchNorm)
        
        # 解码器前向传播，带跳跃连接
        dec1 = self.decoder1(enc8)                    # 512x2x2
        dec1_cat = torch.cat((dec1, enc7), dim=1)     # 1024x2x2
        
        dec2 = self.decoder2(dec1_cat)                # 512x4x4
        dec2_cat = torch.cat((dec2, enc6), dim=1)     # 1024x4x4
        
        dec3 = self.decoder3(dec2_cat)                # 512x8x8
        dec3_cat = torch.cat((dec3, enc5), dim=1)     # 1024x8x8
        
        dec4 = self.decoder4(dec3_cat)                # 512x16x16
        dec4_cat = torch.cat((dec4, enc4), dim=1)     # 1024x16x16
        
        dec5 = self.decoder5(dec4_cat)                # 256x32x32
        dec5_cat = torch.cat((dec5, enc3), dim=1)     # 512x32x32
        
        dec6 = self.decoder6(dec5_cat)                # 128x64x64
        dec6_cat = torch.cat((dec6, enc2), dim=1)     # 256x64x64
        
        dec7 = self.decoder7(dec6_cat)                # 64x128x128
        dec7_cat = torch.cat((dec7, enc1), dim=1)     # 128x128x128
        
        dec8 = self.decoder8(dec7_cat)                # 3x256x256
        
        return dec8