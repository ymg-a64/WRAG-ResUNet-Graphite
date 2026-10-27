import torch
import torch.nn as nn
from model.resnet_backbone import resnet50
from model.modules import ASPP, AttentionGate
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21):
        super(Unet, self).__init__()

        self.resnet = resnet50()
        self.aspp = ASPP(in_channels=2048, out_channels=2048)

        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        #修正：注意力门控的 x_ch 改为实际通道数
        self.att_gate4 = AttentionGate(out_filters[3], 512, out_filters[3] // 2)   # feat3
        self.att_gate3 = AttentionGate(out_filters[2], 256, out_filters[2] // 2)   # feat2
        self.att_gate2 = AttentionGate(out_filters[1], 64, out_filters[1] // 2)    # feat1
        self.att_gate1 = AttentionGate(out_filters[0], 64, out_filters[0] // 2)    # 备用

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        #  feat5 过 ASPP
        feat5 = self.aspp(feat5)

        # 解码 + 注意力门控
        up4 = self.up_concat4(feat4, feat5)
        att_feat3 = self.att_gate4(up4, feat3)
        up3 = self.up_concat3(att_feat3, up4)

        att_feat2 = self.att_gate3(up3, feat2)
        up2 = self.up_concat2(att_feat2, up3)

        att_feat1 = self.att_gate2(up2, feat1)
        up1 = self.up_concat1(att_feat1, up2)

        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        return final
