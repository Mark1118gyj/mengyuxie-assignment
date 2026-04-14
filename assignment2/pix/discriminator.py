import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 判别器，用于 Pix2Pix
    判别器不是判断整张图的真假，而是判断每个 N×N 的 patch 的真假
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        
        # 第一层：不使用 BatchNorm
        sequence = [
            nn.Conv2d(input_nc * 2, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 中间层：逐步增加通道数，减小空间尺寸
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # 倒数第二层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 输出层：1个通道的特征图
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x, y):
        """
        x: 条件图像（输入图像）
        y: 目标图像或生成图像
        """
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)