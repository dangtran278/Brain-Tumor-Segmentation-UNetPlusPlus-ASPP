import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        x = x.float()
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        out += residual  # Add the residual connection
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(
        self, in_channels, out_channels, rates=[1, 3, 6, 9]
    ):  # [1, 3, 6, 9] or [1, 6, 12, 18] or [1, 12, 24, 36]
        super(ASPP, self).__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Atrous convs
        self.atrous_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
                for rate in rates
            ]
        )
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.project = nn.Conv2d(out_channels * 6, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.atrous_convs[0](x)
        x3 = self.atrous_convs[1](x)
        x4 = self.atrous_convs[2](x)
        x5 = self.atrous_convs[3](x)
        x6 = self.global_pool(x)
        x6 = F.interpolate(x6, size=x.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = self.project(x)
        return x


class NestedUNetASPP(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(NestedUNetASPP, self).__init__()
        out_channels = [64, 128, 256, 512, 1024]

        # Backbone
        self.conv0_0 = ResConvBlock(in_channels, out_channels[0], out_channels[0])
        self.conv1_0 = ResConvBlock(out_channels[0], out_channels[1], out_channels[1])
        self.conv2_0 = ResConvBlock(out_channels[1], out_channels[2], out_channels[2])
        self.conv3_0 = ResConvBlock(out_channels[2], out_channels[3], out_channels[3])
        self.conv4_0 = ResConvBlock(out_channels[3], out_channels[4], out_channels[4])

        # Skip pathways
        self.conv0_1 = ResConvBlock(
            out_channels[0] + out_channels[1], out_channels[0], out_channels[0]
        )
        self.conv1_1 = ResConvBlock(
            out_channels[1] + out_channels[2], out_channels[1], out_channels[1]
        )
        self.conv2_1 = ResConvBlock(
            out_channels[2] + out_channels[3], out_channels[2], out_channels[2]
        )
        self.conv3_1 = ResConvBlock(
            out_channels[3] + out_channels[4], out_channels[3], out_channels[3]
        )

        self.conv0_2 = ResConvBlock(
            out_channels[0] * 2 + out_channels[1], out_channels[0], out_channels[0]
        )
        self.conv1_2 = ResConvBlock(
            out_channels[1] * 2 + out_channels[2], out_channels[1], out_channels[1]
        )
        self.conv2_2 = ResConvBlock(
            out_channels[2] * 2 + out_channels[3], out_channels[2], out_channels[2]
        )

        self.conv0_3 = ResConvBlock(
            out_channels[0] * 3 + out_channels[1], out_channels[0], out_channels[0]
        )
        self.conv1_3 = ResConvBlock(
            out_channels[1] * 3 + out_channels[2], out_channels[1], out_channels[1]
        )

        self.conv0_4 = ResConvBlock(
            out_channels[0] * 4 + out_channels[1], out_channels[0], out_channels[0]
        )

        # Pooling, ASPP, Upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.aspp = ASPP(in_channels=1024, out_channels=1024)
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # Classifier
        self.final = nn.Conv2d(out_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Backbone
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.aspp(x4_0)

        # Input = previous conv + upsample lower conv
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_sample(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_sample(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_sample(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up_sample(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_sample(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_sample(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up_sample(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_sample(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up_sample(x2_2)], 1))

        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_sample(x1_3)], 1)
        )
        output = self.final(x0_4)
        return output
