""" ---> moedlin mimarisinin bulunduğu dosya. Bu dosyanın amacı model mimarisinde herhangi bir sorun oluşması dahilinde
kod yapısının uzun satıları arasında sorunu arayıp, yapıyı bozarak değişiklikler yapmak yerine kendi içinde kolayca
düzenlemektir."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""---> temel yapı bloğu. batchnormalizasyonu eğitimi daha hızlı yapabilmek amacıyla aktivasyon fonk. destek için kullandım. 
relu fonksiyonu yaygın kullanılan doğrusal olmayan bir aktivasyon fonk."""

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchN2 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x:torch.Tensor):
        x = self.first(x)
        x = self.batchN2(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)



class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x:torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):
        return self.up(x)



## ANA MIMARI##
"""Girdi olarak görüntüleri alan ve pixellerin hangi sınıflara ait olduğunu tahmmin eden bir segmantasyon çıkartır."""


class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.mid_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x:torch.Tensor):
        skip_connections = []

        for down, pool in zip(self.down_conv, self.down_sample):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.mid_conv(x)

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_sample)):
            x = self.up_sample[idx](x)
            skip = skip_connections[idx]

            if x.shape != skip.shape:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                skip = skip[:, :, diffY // 2: diffY // 2 + x.size(2), diffX // 2: diffX // 2 + x.size(3)]

            x = torch.cat((skip, x), dim=1)
            x = self.up_conv[idx](x)

        x = self.final_conv(x)
        return x

##model giriş ve çıkışlar
model = UNet(3, 59)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)