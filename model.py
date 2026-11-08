"""
model.py - Gelişmiş VTON Generator & Discriminator Mimarisi
- ConvTranspose2d yerine Bilinear Upsampling + Conv2d (Izgara/Moiré desenini %100 yok eder)
- Bottleneck içinde Residual Bloklar (Kumaş dokusunu ve renkleri korur)
- 4 Seviyeli U-Net (256 -> 128 -> 64 -> 32 -> 16)
- Composition Mask (Kişinin yüzünü, saçlarını ve bacaklarını koruyup sadece kıyafet bölgesini giydirir)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Kumaş detaylarını ve renk tutarlılığını koruyan Residual Blok."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpSample(nn.Module):
    """
    Checkerboard/Gökkuşağı ızgara çizgilerini yok eden
    Bilinear Upsampling + Conv2d katmanı.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TryOnGenerator(nn.Module):
    """
    10 Kanal Giriş:
    - agnostic: 3 kanal ([-1, 1])
    - cloth: 3 kanal ([-1, 1])
    - cloth_mask: 1 kanal ([0, 1])
    - pose: 3 kanal ([-1, 1])
    Toplam: 10 kanal
    """
    def __init__(self, in_ch: int = 10, base: int = 64):
        super().__init__()
        # Encoder (256 -> 128 -> 64 -> 32 -> 16)
        self.d1 = DownSample(in_ch, base)            # 256 -> 128, ch=64
        self.d2 = DownSample(base, base * 2)         # 128 -> 64, ch=128
        self.d3 = DownSample(base * 2, base * 4)     # 64 -> 32, ch=256
        self.d4 = DownSample(base * 4, base * 8)     # 32 -> 16, ch=512

        # Bottleneck (Residual Bloklar)
        self.bottleneck = nn.Sequential(
            ResBlock(base * 8),
            ResBlock(base * 8),
            ResBlock(base * 8),
            ResBlock(base * 8),
        )

        # Decoder (16 -> 32 -> 64 -> 128 -> 256)
        self.u4 = UpSample(base * 8, base * 8, base * 4)   # in=512, skip=512, out=256
        self.u3 = UpSample(base * 4, base * 4, base * 2)   # in=256, skip=256, out=128
        self.u2 = UpSample(base * 2, base * 2, base)       # in=128, skip=128, out=64
        self.u1 = UpSample(base, base, base)               # in=64,  skip=64,  out=64

        # Çıkış katmanları:
        # 1. Kıyafet Renk Çıkışı (3 kanal)
        self.out_rgb = nn.Sequential(
            nn.Conv2d(base, 3, kernel_size=3, padding=1),
            nn.Tanh()  # [-1, 1]
        )
        # 2. Akıllı Bileşim Maskesi (1 kanal)
        self.out_mask = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(
        self,
        agnostic: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        pose: torch.Tensor,
        return_mask: bool = False,
    ):
        # Girişleri birleştir (10 kanal)
        x = torch.cat([agnostic, cloth, cloth_mask, pose], dim=1)

        # Encoder
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        # Çıkışlar
        rendered_rgb = self.out_rgb(x)
        comp_mask = self.out_mask(x)

        # Composition: Yüzü/arkaplanı agnostic'ten koru, kıyafeti giydir!
        final_image = comp_mask * rendered_rgb + (1.0 - comp_mask) * agnostic
        if return_mask:
            return final_image, comp_mask
        return final_image


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator (Gerçek/Sahte ve Doku Uyumu)."""
    def __init__(self, in_ch: int = 13):
        # 10 kanal koşul (agnostic, cloth, cloth_mask, pose) + 3 kanal hedef/fake
        super().__init__()
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_ch, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, agnostic, cloth, cloth_mask, pose, target):
        x = torch.cat([agnostic, cloth, cloth_mask, pose, target], dim=1)
        return self.model(x)