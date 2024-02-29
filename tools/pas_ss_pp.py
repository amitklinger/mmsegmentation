# Copyright (c) Hailo Inc. All rights reserved.
import torch
import torch.nn as nn
import numpy as np


class PostProcess(nn.Module):
    def __init__(self, num_classes,  **kwargs):
        super(PostProcess, self).__init__(**kwargs)
        self.num_classes = num_classes

        self.dw = torch.nn.Conv2d(in_channels=self.num_classes - 1, out_channels=self.num_classes - 1, kernel_size=(1, 2), groups=self.num_classes - 1, bias=False)
        w = np.ones((self.num_classes - 1,1,1,2), dtype=np.float32)
        w[:, :, :, 1] = -1
        self.dw.weight = torch.nn.Parameter(torch.Tensor(w))
        self.relu = torch.nn.ReLU()
        self.argmax_star = True  #  Argmax* (Argmax from bottom) -> will not compile, so default is False

    def forward(self, x):
        # input is (BxCxHxW): 1x10x92x120 output is 1x10x736x240
        x = torch.nn.functional.interpolate(x, size=(736, 240), mode='bilinear', align_corners=True)

        # argmax on channels. output is 1x1x736x240
        x = torch.argmax(x, dim=1, keepdim=True)

        # H<->W transpose. output is 1x1x240x736
        x = torch.transpose(x, 2, 3)

        # torch.nn.functional.one_hot adds an extra dim at the end of the tensor so output is 1x1x240x736x10
        x = torch.nn.functional.one_hot(x, num_classes=self.num_classes)  
        x = torch.transpose(x, 1, 4)  # output is 1x10x240x736x1
        x = torch.squeeze(x, dim=-1)  # output is 1x10x240x736

        # First output edge detector
        out1 = x[:, :-1, :, :]  # output is 1x9x240x736. Assuming raindrop is last class
        # out1 = torch.nn.functional.pad(out1, [1, 0, 0, 0])  # output is 1x9x240x737
        out1 = out1.to(torch.float32)
        out1 = torch.nn.functional.pad(out1, [0, 1, 0, 0], mode='constant', value=0.5)  # output is 1x9x240x737
        out1 = self.relu(self.dw(out1))  # output is 1x9x240x736

        # W<->C transpose. output is 1x736x240x9
        out1 = torch.transpose(out1, 1, 3)

        if self.argmax_star:
            # argmax* support: Flip the 736 axis. output is 1x736x240x9
            out1 = torch.flip(out1, dims=(1,))
        # argmax on channels. final output is 1x1x240x9
        out1 = torch.argmax(out1, dim=1, keepdim=True)

        # second output is to reduce sum on final class. output is 4 integers of 1x4x1x1 so each one would be represented by 16 bit integer. Assuming raindrop is last class
        # x is of shape = [1, 1, 240, 736]
        xr = x[:, -1:, :, :]
        xr = xr[:, 0, :, :]
        # reduce1, reduce2, reduce3, reduce4, reduce5, reduce6 = xr[:, :40, :], xr[:, 40:80, :], xr[:, 80:120, :], xr[:, 120:160, :], xr[:, 160:200, :], xr[:, 200:, :]
        reduce1, reduce2, reduce3, reduce4, reduce5, reduce6, reduce7, reduce8 = xr[:, :30, :], xr[:, 30:60, :], xr[:, 60:90, :], xr[:, 90:120, :], xr[:, 120:150, :], xr[:, 150:180, :], xr[:, 180:210, :], xr[:, 210:, :]
        rsum1 = torch.sum(torch.sum(reduce1, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum2 = torch.sum(torch.sum(reduce2, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum3 = torch.sum(torch.sum(reduce3, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum4 = torch.sum(torch.sum(reduce4, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum5 = torch.sum(torch.sum(reduce5, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum6 = torch.sum(torch.sum(reduce6, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum7 = torch.sum(torch.sum(reduce7, dim=-1, keepdim=True), dim=-2, keepdim=True)
        rsum8 = torch.sum(torch.sum(reduce8, dim=-1, keepdim=True), dim=-2, keepdim=True)
        out2 = torch.concat([rsum1, rsum2, rsum3, rsum4, rsum5, rsum6, rsum7, rsum8], dim=2)

        return out1, out2