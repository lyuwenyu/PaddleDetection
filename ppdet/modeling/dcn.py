import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.ops import DeformConv2D

import numpy as np


class DCN2D(nn.Layer):
    def __init__(
            self,
            in_c,
            out_c,
            kernel,
            stride=1, ):
        super().__init__()

        self.offset_channel = 2 * kernel**2
        self.mask_channel = kernel**2

        self.conv_offset = nn.Conv2D(
            in_channels=in_c,
            out_channels=3 * kernel**2,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel - 1) // 2, )

        self.conv = DeformConv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel - 1) // 2,
            dilation=1,
            bias_attr=False)

    def forward(self, x):

        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        out = self.conv(x, offset, mask=mask)

        return out


class MSDCN2D(nn.Layer):
    def __init__(
            self,
            in_c,
            out_c,
            kernels,
            stride=1, ):
        super().__init__()
        self.kernels = kernels

        kernel = sum([k**2 for k in kernels])
        self.offset_channel = 2 * kernel
        self.mask_channel = kernel

        self.conv_offset = nn.Conv2D(
            in_channels=in_c,
            out_channels=self.offset_channel + self.mask_channel,
            kernel_size=kernels[0],
            stride=stride,
            padding=(kernels[0] - 1) // 2, )

        self.convs = nn.LayerList([
            DeformConv2D(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k,
                stride=stride,
                padding=(k - 1) // 2,
                dilation=1,
                bias_attr=False) for k in kernels
        ])

        self.weights = nn.Embedding(len(kernels), 1)
        self.weights.weight.set_value(np.ones(len(kernels), 1, dtype='float32'))

    def forward(self, x, feats):

        offset_mask = self.conv_offset(x)
        offsets, masks = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        masks = F.sigmoid(masks)

        offset_idx = 0
        mask_idx = 0

        outputs = []
        for i, y in enumerate(feats):
            _offset_idx = offset_idx + self.kernels[i]**2 * 2
            _mask_idx = mask_idx + self.kernels[i]**2

            _, _, h, w = offsets.shape
            y = F.interpolate(y, size=(h, w))

            # print(i, y.shape)
            # print(f' {idx} : {_idx} ', offsets[:, idx:_idx].shape)
            # print(masks[:, i:i+1].shape)
            # print('y ', y.shape)

            out = self.convs[i](y,
                                offsets[:, offset_idx:_offset_idx],
                                mask=masks[:, mask_idx:_mask_idx])
            outputs.append(out)

            offset_idx = _offset_idx
            mask_idx = _mask_idx

        out = 0
        for i, o in enumerate(outputs):
            out += o * self.weights[i]

        # out = sum(outputs)

        return out


class MSDCNHead(nn.Layer):
    def __init__(self, hidden_dim, kernels=[
            3,
            3,
            3,
    ], use_pan=False):
        super().__init__()

        self.kernels = kernels
        self.num_levels = len(kernels)

        self.fpns = nn.LayerList([
            nn.Conv2DTranspose(
                hidden_dim, hidden_dim, kernel_size=2, stride=2), nn.Identity(),
            nn.MaxPool2D(2, 2)
        ])

        self.dcns = nn.LayerList(
            [MSDCN2D(hidden_dim, hidden_dim, kernels) for _ in kernels])

        # if use_pan:
        #     from ppdet.modeling.necks import YOLOCSPPAN
        #     self.pan = YOLOCSPPAN([hidden_dim for _ in range(3)])

    def forward(self, feats):
        assert len(feats) == self.num_levels, ''

        preds = [m(feats[-1]) for m in self.fpns]
        preds = [m(x, feats) for m, x in zip(self.dcns, preds)]

        return preds


if __name__ == '__main__':

    feats = [paddle.rand([1, 5, s, s]) for s in [10, 10, 10, 10]]
    m = MSDCNHead(5, [3, 3, 3, 3])

    print(m)

    print([o.shape for o in m(feats)])
