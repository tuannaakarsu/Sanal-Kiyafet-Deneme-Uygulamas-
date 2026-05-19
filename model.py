import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return feat, self.pool(feat)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
    
class TryOnGenerator(nn.Module):
    """
    Giris kanallari:
    - agnostic: 3
    - cloth: 3
    - cloth_mask: 1
    - pose: 3
    Toplam: 10 kanal
    """

    def __init__(self, in_ch: int = 10, base: int = 64):
        super().__init__()
        self.d1 = Down(in_ch, base)
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3 = Up(base * 8, base * 4)
        self.u2 = Up(base * 4, base * 2)
        self.u1 = Up(base * 2, base)
        self.out = nn.Conv2d(base, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        agnostic: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([agnostic, cloth, cloth_mask, pose], dim=1)
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        x = self.mid(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.out(x)
        return self.tanh(x)
    