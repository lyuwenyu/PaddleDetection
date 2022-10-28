import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.ops import DeformConv2D


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

        kernel = sum(kernels)

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

    def forward(self, x, feats):

        offset_mask = self.conv_offset(x)
        offsets, masks = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        masks = F.sigmoid(masks)

        idx = 0
        outputs = []
        for i, y in enumerate(feats):
            _idx = idx + self.kernels[i]**2
            out = self.conv(y, offsets[idx:_idx], mask=masks)
            outputs.append(out)

            idx = _idx

        out = sum(outputs)

        return out
