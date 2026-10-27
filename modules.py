import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1)
        self.branch2 = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d) for d in dilations
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.conv_out = nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = []
        features.append(self.branch1(x))
        for branch in self.branch2:
            features.append(branch(x))
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        features.append(global_feat)
        out = torch.cat(features, dim=1)
        out = self.conv_out(out)
        out = self.bn(out)
        return self.relu(out)
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.W_g = nn.Conv2d(g_ch, int_ch, 1)
        self.W_x = nn.Conv2d(x_ch, int_ch, 1)
        self.psi = nn.Conv2d(int_ch, 1, 1)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        attn = torch.sigmoid(self.psi(torch.relu(g_conv + x_conv)))
        return x * attn + x
