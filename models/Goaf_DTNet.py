# add channel attention module in the feature fusion module connnecting the encoder and segmentaion decoder

import torch.nn as nn
import torch
from models.resnet import Bottleneck, BasicBlock
from models.modules import SEBlock
import torch.nn.functional as F


class GoafDTNet(nn.Module):
    def __init__(self, cfg):
        super(GoafDTNet, self).__init__()
        self.cfg = cfg

        self.backbone = Backbone(BasicBlock, blocks_num=[3,4,6,3], num_classes=1, cfg=cfg)
        self.segmentation_decoder4 = SegDecoder(in_channels=512, stage="segmentation_decoder", deep_supervision_stage="segmentation_ds", cfg=cfg)
        self.segmentation_decoder3 = SegDecoder(in_channels=256, stage="segmentation_decoder", deep_supervision_stage="segmentation_ds", cfg=cfg)
        self.segmentation_decoder2 = SegDecoder(in_channels=128, stage="segmentation_decoder", deep_supervision_stage="segmentation_ds", cfg=cfg)
        self.segmentation_decoder1 = SegDecoder(in_channels=64, stage="segmentation_decoder", deep_supervision_stage="segmentation_ds", cfg=cfg)
        self.segmentation_conv_block = ConvBlock(32)
        self.segmentation_decoders = [
            self.segmentation_decoder1, self.segmentation_decoder2,
            self.segmentation_decoder3, self.segmentation_decoder4,
        ]

        self.sgfm_4 = SGFM(512)
        self.sgfm_3 = SGFM(256)
        self.sgfm_2 = SGFM(128)
        self.sgfm_1 = SGFM(64)
        self.sgfms = [self.sgfm_1, self.sgfm_2, self.sgfm_3, self.sgfm_4]

        self.extraction_decoder4 = ExtractionDecoder(512, stage="extraction_decoder", deep_supervision_stage="extraction_ds", cfg=cfg)
        self.extraction_decoder3 = ExtractionDecoder(256, stage="extraction_decoder", deep_supervision_stage="extraction_ds", cfg=cfg)
        self.extraction_decoder2 = ExtractionDecoder(128, stage="extraction_decoder", deep_supervision_stage="extraction_ds", cfg=cfg)
        self.extraction_decoder1 = ExtractionDecoder(64, stage="extraction_decoder", deep_supervision_stage="extraction_ds", cfg=cfg)
        self.extraction_conv_block = ConvBlock(32)
        self.extraction_decoders = [
            self.extraction_decoder1, self.extraction_decoder2,
            self.extraction_decoder3, self.extraction_decoder4
        ]

        self.conv1 = nn.Conv2d(5, 1, 1)
        self.conv2 = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        """
        :param x: Raw input image
        :return:
        """

        segmentation_head_features = {}
        stage_index = 5
        x = self.backbone(x)
        x_aspp =x[5]

        for i in sorted(range(stage_index), reverse=True):
            if i != 0:
                seg_decoder_input = [x[i-1], x[i], x[5]]
                segmentation_head_output = self.segmentation_decoders[i-1](seg_decoder_input)
                x_segmentation_decoder, x_segmentation_ds = segmentation_head_output["segmentation_decoder"], segmentation_head_output["segmentation_ds"]
                x[5] = x_segmentation_decoder
                segmentation_head_features["segmentation_features_decoder{}".format(str(i))] = x_segmentation_decoder
                segmentation_head_features["segmentation_features_ds{}".format(str(i))] = x_segmentation_ds
            # last convolution module in the segmentation head decoder
            else:
                segmentation_end_out = self.segmentation_conv_block(x[5])

        stage_index = 4
        SGFM_features = {}
        for i in sorted(range(1, stage_index+1), reverse=True):
            extraction_sgfm_input = [
                x[i-1],
                x[i],
                segmentation_head_features["segmentation_features_ds{}".format(str(i))],
            ]

            SGFM_features["SGFM_stage{}".format(str(i))] = self.sgfms[i-1](extraction_sgfm_input)

        stage_index = 5
        extraction_head_features = {}
        for i in sorted(range(stage_index), reverse=True):
            if i!= 0:
                extraction_decoder_input = [
                    SGFM_features["SGFM_stage{}".format(str(i))],
                    x_aspp,
                ]
                extraction_head_output = self.extraction_decoders[i-1](extraction_decoder_input[0], extraction_decoder_input[1])
                x_extraction_decoder, x_extraction_ds = extraction_head_output["extraction_decoder"], extraction_head_output["extraction_ds"]
                extraction_head_features["extraction_features_decoder{}".format(str(i))] = x_extraction_decoder
                extraction_head_features["extraction_features_ds{}".format(str(i))] = x_extraction_ds
                x_aspp = x_extraction_decoder
            else:
                extraction_end_output = self.extraction_conv_block(x_aspp)


        # aggregate all the feature maps
        segmentation_out = {}
        extraction_out = {}
        stage_index = 5
        for i in range(stage_index):
            if i == 0:
                segmentation_out["end_out"] = segmentation_end_out
                extraction_out["end_out"] = extraction_end_output
                # print(i, "segmentation_out: ", segmentation_end_out.shape)
                # print(i, "extraction_out: ", extraction_end_output.shape)
            else:   # Upsample output of the middle stage from both the segmentation and extraction head decoder for deep supervision
                segmentation_out["ds_out{}".format(str(i))] = F.interpolate(segmentation_head_features["segmentation_features_ds{}".format(str(i))], scale_factor=2**i, mode="bilinear")
                extraction_out["ds_out{}".format(str(i))] = F.interpolate(extraction_head_features["extraction_features_ds{}".format(str(i))], scale_factor=2**i, mode="bilinear")
        segmentation_out['end_out'] = self.conv1(torch.cat((segmentation_out['end_out'],segmentation_out["ds_out1"], segmentation_out["ds_out2"], segmentation_out["ds_out3"], segmentation_out["ds_out4"]), dim=1))
        extraction_out['end_out'] = self.conv2(torch.cat((extraction_out['end_out'], extraction_out['ds_out1'], extraction_out["ds_out2"], extraction_out["ds_out3"], extraction_out["ds_out4"]), dim=1))
        return segmentation_out, extraction_out


# backbone of the network, stacking basicblock + channel attention
class Backbone(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=False, groups=1, width_per_group=64, cfg=None):
        super(Backbone, self).__init__()
        self.cfg = cfg

        self.include_top = include_top
        self.in_channel = 32

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.cfg.MODEL.ASPP == True:
            self.aspp = ASPP()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        s0 = x
        out.append(s0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        s1 = self.layer1(x)
        out.append(s1)
        # ca1 = self.calayer1(s1)
        s2 = self.layer2(s1)
        out.append(s2)
        # ca2 = self.calayer2(s2)
        s3 = self.layer3(s2)
        out.append(s3)
        # ca3 = self.calayer3(s3)
        s4 = self.layer4(s3)
        out.append(s4)
        # ca4 = self.calayer4(s4)
        if self.cfg.MODEL.ASPP == True:
            y = self.aspp(s4)
            out.append(y)
        else:
            out.append(s4)

        if self.include_top:
            x = self.avgpool(s4)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()
        """
        Convolution Block at the end of task-specific decoder 
        """

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, 1, 1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels*2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1, 1),
        )

    def forward(self, x):
        """
        :param x: feaure maps from the last decoder stage
        :return:
        """
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return self.conv(x)


def conv3x3(in_channels, out_channels):
    conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d( out_channels),
            nn.ReLU(),
        )
    return conv


class SegDecoder(nn.Module):
    def __init__(self, in_channels, stage="segmentation_decoder", deep_supervision_stage="seg_ds", cfg=None):
        super(SegDecoder, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.ModuleList([conv3x3(in_channels, in_channels//2), conv3x3(in_channels//2, in_channels)])     # 512 -> 256 -> 512
        self.conv2 = nn.Conv2d(in_channels, in_channels//2, 1, 1)
        self.conv3 = conv3x3(in_channels, in_channels)
        self.conv4 = conv3x3(in_channels, in_channels)
        self.conv5 = nn.Conv2d(in_channels, in_channels//2, 1, 1)

        self.calayer = SEBlock(in_channels)

        # convolution layers for deep supervision in segmentation head decoder
        self.ds = nn.ModuleList([conv3x3(in_channels, 64), nn.Conv2d(64, 1, 1, 1)])

        #
        self.stage = stage
        self.deep_supervision_stage = deep_supervision_stage

    def forward(self, x):
        """
        :param x: list contains features from the backbone
        :return:
        """
        out = {}
        x1 = x[0]   # feature maps from the lower stage in the encoder
        x2 = x[1]   # feature maps from the corresponding encoder stage
        y = x[2]    # feature maps from last segmentation decoder or ASPP module
        if self.cfg.MODEL.MFFM == True:
            f = self.feature_fusion(x2, y, x1)
        else:
            f = F.interpolate(y, scale_factor=2, mode="bilinear")
        f = self.conv4(f)
        out[self.stage] = self.conv5(f)
        for m in self.ds:
            f = m(f) # features for deep supervision
        out[self.deep_supervision_stage] = f

        return out

    def feature_fusion(self, x1, x2, x3):
        """
        :param x1: feature maps from the corresponding encoder stage
        :param x2: feature maps from last segmentation decoder or ASPP module
        :return:x3: feature maps from the lower stage in the encoder
        """
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear")
        x1 = self.conv2(x1)
        y = torch.cat((x1, x3), dim=1)
        y = self.conv3(y)
        y = self.calayer(y)

        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        for m in self.conv1:
            x2 = m(x2)

        features = y + x2

        return features


class SGFM(nn.Module):
    def __init__(self, in_channels):
        super(SGFM, self).__init__()

        self.conv1 = nn.ModuleList([conv3x3(in_channels, in_channels//2), conv3x3(in_channels//2, in_channels//2)])
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels//2, 1, 1)
        )

    def forward(self, x):
        """
        :param x: list contains feature maps
        :return:
        """
        x1 = x[0]  # feature maps from the  lower stage of the backbone
        x2 = x[1]  # backbone feature maps from corresponding extraction stage in the backbone
        x3 = x[2]  # feature maps(channel dimension equals 1) from the corresponding stage in the segmentation decoder
        x3 = torch.sigmoid(x3)

        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        for m in self.conv1:
            x2 = m(x2)

        # y = x2 + x1
        y = torch.cat((x2, x1), dim=1)
        y = self.conv2(y)
        y = torch.mul(y, x3)
        y = self.conv3(y)

        return y


class ExtractionDecoder(nn.Module):
    def __init__(self, in_channels, stage="extraction_decoder", deep_supervision_stage="extraction_ds", cfg=None):
        super(ExtractionDecoder, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        if self.cfg.MODEL.SGFM == True:
            self.conv2 = conv3x3(in_channels, in_channels//2)
        else:
            self.conv2 = conv3x3(in_channels, in_channels)
        self.conv3 = nn.ModuleList([conv3x3(in_channels, in_channels//2), conv3x3(in_channels//2, in_channels)])
        self.conv4 = nn.Conv2d(in_channels, in_channels//2, 1, 1)

        # convolution layers for deep supervision
        self.ds = nn.ModuleList([conv3x3(in_channels, 64), nn.Conv2d(64, 1, 1, 1) ])

        #
        self.stage = stage
        self.deep_supervision_stage = deep_supervision_stage

    def forward(self, x, y):
        """
        :param x: feature maps from the corresponding segmentation-guided feature module
        :param y: feature maps from the last extraction decoder or ASPP module
        :return:
        """
        out = {}
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        y = self.conv1(y)
        y = self.conv2(y)
        if self.cfg.MODEL.SGFM == True:
            f = torch.cat((x, y), dim=1)
        else:
            f = y

        for m in self.conv3:
            f = m(f)

        ds = f
        for m in self.ds:
            ds = m(ds)
        out[self.stage] = self.conv4(f)
        out[self.deep_supervision_stage] = ds

        return out


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.relu = nn.ReLU()

        self.conv_1x1_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)

        self.conv_3x3_1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(128)

        self.conv_3x3_2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv3_3x3_2 = nn.BatchNorm2d(128)

        self.conv_3x3_3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv3_3x3_3 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_2 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn_conv1x1_2 = nn.BatchNorm2d(128)

        self.conv1x1_3 = nn.Conv2d(128*5, 512, kernel_size=1)
        self.bn_conv1x1_3 = nn.BatchNorm2d(512)



    def forward(self,x):
        h = x.size()[2]
        w = x.size()[3]
        # print(x.shape)


        out_1x1 = self.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        out_3x3_1 = self.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        out_3x3_2 = self.relu(self.bn_conv3_3x3_2(self.conv_3x3_2(x)))
        out_3x3_3= self.relu(self.bn_conv3_3x3_3(self.conv_3x3_3(x)))


        # image pooling
        out_img = self.avg_pool(x)
        out_img = self.relu(self.bn_conv1x1_2(self.conv1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(h,w),mode="bilinear")

        out = torch.cat([out_1x1,out_3x3_1,out_3x3_2,out_3x3_3,out_img], dim=1)
        out = self.relu(self.bn_conv1x1_3(self.conv1x1_3(out)))

        return out


if __name__ == "__main__":
    from configs.defaults import _C as cfg
    cfgFile = r"./configs/mffdn.yaml"
    cfg.merge_from_file(cfgFile)
    print(cfg)
    import numpy as np
    x = torch.randn((2,3,256,256))
    print(x.shape)
    model = GoafDTNet(cfg)
    out = model(x)
    print(len(out), out[0]["end_out"].shape)
