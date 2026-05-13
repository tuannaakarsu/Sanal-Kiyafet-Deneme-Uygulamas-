import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
####################################################
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.2):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchN2 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.dropout = nn.Dropout2d(p=dropout_rate)  # Dropout2d tercih edilir

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.batchN2(x)
        x = self.act1(x)
        x = self.dropout(x)  # Dropout burada çalışır
        x = self.second(x)
        return self.act2(x)
##############################################################
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x:torch.Tensor):
        return self.pool(x)
##################################################################
class UpSample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor):
        return self.up(x)
##############################################################################
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.down_conv = nn.ModuleList([
            DoubleConvolution(in_channels, 64),
            DoubleConvolution(64, 128),
            DoubleConvolution(128, 256),
        ])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(3)])

        self.mid_conv = DoubleConvolution(256, 512)

        self.up_sample = nn.ModuleList([
            UpSample(512, 256),
            UpSample(256, 128),
            UpSample(128, 64),
        ])
        self.up_conv = nn.ModuleList([
            DoubleConvolution(512, 256),
            DoubleConvolution(256, 128),
            DoubleConvolution(128, 64),
        ])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
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
###########################################################################
def edge_loss(pred, target, edge_weight=1.0):
    # Sobel filtreleri
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    # Tahmin edilen kenar haritası
    pred_soft = F.softmax(pred, dim=1)
    pred_edges = torch.abs(F.conv2d(pred_soft[:, 1:2], sobel_x, padding=1)) + \
                 torch.abs(F.conv2d(pred_soft[:, 1:2], sobel_y, padding=1))

    # Gerçek maskeyi one-hot yap ve kenarları al
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    target_edges = torch.abs(F.conv2d(target_onehot[:, 1:2], sobel_x, padding=1)) + \
                   torch.abs(F.conv2d(target_onehot[:, 1:2], sobel_y, padding=1))

    # Kenar farkları
    return F.l1_loss(pred_edges, target_edges) * edge_weight


def dice_loss(pred, target, smooth=1e-3):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def total_segmentation_loss(outputs, masks, weight_ce=1.0, weight_dice=1.0, weight_edge=0.6):
    # Cross Entropy Loss
    ce = F.cross_entropy(outputs, masks)

    # Dice için hazırlık
    masks_onehot = F.one_hot(masks, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
    probs = F.softmax(outputs, dim=1)
    d_loss = dice_loss(probs, masks_onehot)

    # Edge-aware loss
    e_loss = edge_loss(outputs, masks, edge_weight=1.0)

    # Toplam
    total = weight_ce * ce + weight_dice * d_loss + weight_edge * e_loss
    return total
##############################################################
model = UNet(3, 59)
################################################################
criterion = total_segmentation_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
