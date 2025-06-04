import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, dropout=0, reg=0.01):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=4, dropout=0.18, dilation_rate=1, reg=0.01):
        super().__init__()

        self.inc = DoubleConv(input_channels, 32, dilation_rate, dropout, reg)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64, dilation_rate, dropout, reg)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128, dilation_rate, dropout, reg)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256, dilation_rate, dropout, reg)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512, dilation_rate, dropout, reg)
        )

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256, dilation_rate, dropout, reg)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128, dilation_rate, dropout, reg)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64, dilation_rate, dropout, reg)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(64, 32, dilation_rate, dropout, reg)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)

        x = self.outc(x)
        return F.softmax(x, dim=1)


class WeightedCategoricalCrossEntropyLoss(nn.Module):
    def __init__(self, weights=[1., 1., 1., 1.]):
        super().__init__()
        self.weights = torch.tensor(weights).float()
        print('The used loss function is: weighted categorical crossentropy')

    def forward(self, y_pred, y_true):
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()
        
        # 确保权重维度与预测维度匹配
        weights = self.weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
        
        # Calculate categorical crossentropy
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        loss = -torch.sum(y_true * torch.log(y_pred), dim=1)  # [N, H, W]
        
        # 计算每个像素的权重
        weights_sum = torch.sum(y_true * weights, dim=1)  # [N, H, W]
        
        # 应用权重并计算平均损失
        weighted_loss = loss * weights_sum
        return torch.mean(weighted_loss)