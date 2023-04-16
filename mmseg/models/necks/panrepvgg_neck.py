# Hailo #

import torch.nn as nn
import numpy as np
import torch
from ..backbones.repvgg import RepVGGBlock


class ConvBNAct(nn.Module):  # TODO: Amit -- can replace with `BaseConv` by YOLOX
    '''Conv2d + BN + Activation (ReLU)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Transpose(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample_transpose = torch.nn.UpsamplingNearest2d(scale_factor=scale_factor)
        # self.upsample_transpose = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample_transpose(x)


def make_repvgg_stage(in_channels, out_channels, num_blocks, stride=1, deploy=False):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    for stride in strides:
        blocks.append(RepVGGBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=stride, padding=1, groups=1, deploy=deploy, use_se=False))
        in_channels = out_channels
    return nn.Sequential(*blocks)


class PANRepVGGNeck(nn.Module):
    def __init__(
        self,
        num_repeats=None,  # [12, 12, 12, 12]
        width_multiplier=None,  # [0.25, 0.25, 0.25, 0.25]
        channels_list=None
        ):

        super().__init__()
        neck_num_repeats = [int(r * w) for r, w in (zip(num_repeats, width_multiplier))]  # [3, 3, 3, 3]

        assert channels_list is not None
        assert num_repeats is not None


        '''
        ################################### org size #######################################################
        self.neck_p4 = make_repvgg_stage(192, 64, neck_num_repeats[0])
        self.neck_p3 = make_repvgg_stage(96, 32, neck_num_repeats[1])
        self.neck_n3 = make_repvgg_stage(64, 64, neck_num_repeats[2])
        self.neck_n4 = make_repvgg_stage(128, 128, neck_num_repeats[3])

        self.align_channels0 = ConvBNAct(in_channels=256, out_channels=64, kernel_size=1)
        self.upsample0 = Transpose()
        self.align_channels1 = ConvBNAct(in_channels=64, out_channels=32, kernel_size=1)
        self.upsample1 = Transpose()
        self.downsample2 = nn.Sequential(
            ConvBNAct(in_channels=32, out_channels=32, kernel_size=3),  # Added to Tal's request
            ConvBNAct(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        )
        self.downsample1 = ConvBNAct(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        ########################################################################################################
        '''

        ##################################### SS size ###########################################
        self.neck_p4 = make_repvgg_stage(96, 32, neck_num_repeats[0])
        self.neck_p3 = make_repvgg_stage(48, 16, neck_num_repeats[1])
        self.neck_n3 = make_repvgg_stage(32, 32, neck_num_repeats[2])
        self.neck_n4 = make_repvgg_stage(64, 64, neck_num_repeats[3])

        self.align_channels0 = ConvBNAct(in_channels=128, out_channels=32, kernel_size=1)
        self.upsample0 = Transpose()
        self.align_channels1 = ConvBNAct(in_channels=32, out_channels=16, kernel_size=1)
        self.upsample1 = Transpose()
        self.downsample2 = nn.Sequential(
            ConvBNAct(in_channels=16, out_channels=16, kernel_size=3),  # Added to Tal's request
            ConvBNAct(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        )
        self.downsample1 = ConvBNAct(in_channels=32, out_channels=32, kernel_size=3, stride=2)


    def forward(self, input):
        """
        Args:
            input (Tuple[Tensor]): backbone output features:
                outputs[0] - /8 resolution feature map
                outputs[1] - /16 resolution feature map
                outputs[2] - /32 resolution feature map
        Returns:
            List[Tensor]: neck output features:
                outputs[0] - /8 resolution feature map
                outputs[1] - /16 resolution feature map
                outputs[2] - /32 resolution feature map
        """
        (x2, x1, x0) = input
        fpn_out0 = self.align_channels0(x0)
        upsample_features0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_features0, x1], 1)
        f_out0 = self.neck_p4(f_concat_layer0)

        fpn_out1 = self.align_channels1(f_out0)
        upsample_features1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_features1, x2], 1)
        pan_out2 = self.neck_p3(f_concat_layer1)  # 92,120,48 -> 92,120,16

        down_features1 = self.downsample2(pan_out2) # 92,120,16 -> 46,60,16
        p_concat_layer1 = torch.cat([down_features1, fpn_out1], 1)  # 46,60,16 cat 46,60,16 -> 46,60,32
        pan_out1 = self.neck_n3(p_concat_layer1)  # 46,60,32 -> 46,60,32

        down_features0 = self.downsample1(pan_out1)  # 23,30,32
        p_concat_layer2 = torch.cat([down_features0, fpn_out0], 1) # 23,30,32 cat 23,30,32 -> 23,30,64
        pan_out0 = self.neck_n4(p_concat_layer2) # 23,30,64 -> 23,30,34

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs