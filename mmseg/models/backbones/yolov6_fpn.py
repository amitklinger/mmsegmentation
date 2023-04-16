#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Hailo Inc. All rights reserved. TODO: Amit -- is that OK?

import torch
import torch.nn as nn

from .repvgg import RepVGG
from ..necks import PANRepVGGNeck
from .network_blocks import BaseConv
from ..builder import BACKBONES


@BACKBONES.register_module()
class YOLOv6FPN(nn.Module):
    """
    YOLOv6FPN module.
    """

    def __init__(
        self,
        depth = 1.0,
        width = 1.0,
        act="silu",
    ):
        super().__init__()

        self.channels_list = [32, 64, 128, 256]  # , 64, 32, 32, 64, 64, 128]

        #self.num_repeats = [3, 4, 6, 3]  # org
        self.num_repeats = [3, 5, 7, 4]   # plus 1
        width_multiplier = [0.125] * len(self.channels_list)

        self.backbone = RepVGG(num_blocks=self.num_repeats, width_multiplier=width_multiplier)

        #self.neck_num_repeats = [12, 12, 12, 12]   # org
        self.neck_num_repeats = [12, 16, 16, 12]    # plus 1
        self.neck_out_channels = [256, 128, 128, 256, 256, 512]
        width_multiplier = [0.25] * len(self.channels_list)
        self.neck = PANRepVGGNeck(num_repeats=self.neck_num_repeats, width_multiplier=width_multiplier, channels_list=self.neck_out_channels)


    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.
        Returns:
            Tuple[Tensor]: FPN output features.
        """
        # backbone
        backbone_features = self.backbone(inputs)

        # neck
        neck_features = self.neck(backbone_features)
        return neck_features