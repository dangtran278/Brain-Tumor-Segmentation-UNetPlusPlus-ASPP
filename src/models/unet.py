import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(UNet, self).__init__()
        out_channels = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv0 = ConvBlock(in_channels, out_channels[0])
        self.conv1 = ConvBlock(out_channels[0], out_channels[1])
        self.conv2 = ConvBlock(out_channels[1], out_channels[2])
        self.conv3 = ConvBlock(out_channels[2], out_channels[3])
        self.conv4 = ConvBlock(out_channels[3], out_channels[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upsample4 = UpsampleBlock(out_channels[4], out_channels[3])
        self.up_conv4 = ConvBlock(out_channels[4], out_channels[3])

        self.upsample3 = UpsampleBlock(out_channels[3], out_channels[2])
        self.up_conv3 = ConvBlock(out_channels[3], out_channels[2])

        self.upsample2 = UpsampleBlock(out_channels[2], out_channels[1])
        self.up_conv2 = ConvBlock(out_channels[2], out_channels[1])

        self.upsample1 = UpsampleBlock(out_channels[1], out_channels[0])
        self.up_conv1 = ConvBlock(out_channels[1], out_channels[0])

        self.final = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.conv0(x)
        e2 = self.conv1(self.pool(e1))
        e3 = self.conv2(self.pool(e2))
        e4 = self.conv3(self.pool(e3))
        e5 = self.conv4(self.pool(e4))

        d5 = self.up_conv4(torch.cat([e4, self.upsample4(e5)], 1))
        d4 = self.up_conv3(torch.cat([e3, self.upsample3(d5)], 1))
        d3 = self.up_conv2(torch.cat([e2, self.upsample2(d4)], 1))
        d2 = self.up_conv1(torch.cat([e1, self.upsample1(d3)], 1))
        output = self.final(d2)

        return output
