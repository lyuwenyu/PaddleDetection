# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
# 
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_
from ..shape_spec import ShapeSpec

__all__ = [
    'CSPDarkNet', 'BaseConv', 'DWConv', 'BottleNeck', 'SPPLayer', 'SPPFLayer'
]


class BaseConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        from typing import Callable
        if isinstance(act, Callable):
            self.act = act
        else:
            if act == 'silu' or act == 'swish':
                self.act = lambda x: x * F.sigmoid(x)
            else:
                self.act = getattr(F, act)

        # print(self.act)

        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        # y = x * F.sigmoid(x)
        y = self.act(x)

        return y


class DWConv(nn.Layer):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act)
        self.pw_conv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            bias=bias,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class Focus(nn.Layer):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(
            in_channels * 4,
            out_channels,
            ksize=ksize,
            stride=stride,
            bias=bias,
            act=act)

    def forward(self, inputs):
        # inputs [bs, C, H, W] -> outputs [bs, 4C, W/2, H/2]
        top_left = inputs[:, :, 0::2, 0::2]
        top_right = inputs[:, :, 0::2, 1::2]
        bottom_left = inputs[:, :, 1::2, 0::2]
        bottom_right = inputs[:, :, 1::2, 1::2]
        outputs = paddle.concat(
            [top_left, bottom_left, top_right, bottom_right], 1)
        return self.conv(outputs)


from ppdet.modeling.ops import get_act_fn
from paddle.nn.initializer import Constant


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class RepVggBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)

        # self.act = get_act_fn(act) if act is None or isinstance(act, (
        #     str, dict)) else act

        from typing import Callable
        if isinstance(act, Callable):
            self.act = act
        else:
            if act == 'silu' or act == 'swish':
                self.act = lambda x: x * F.sigmoid(x)
            else:
                self.act = getattr(F, act)

        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BottleNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu",
                 use_repconv=False):
        super(BottleNeck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv

        # if use_repconv and hidden_channels == hidden_channels:
        #     self.conv1 = nn.Identity()
        # else:
        #     self.conv1 = BaseConv(
        #         in_channels,
        #         hidden_channels,
        #         ksize=1,
        #         stride=1,
        #         bias=bias,
        #         act=act)

        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        if use_repconv:
            self.conv2 = RepVggBlock(
                hidden_channels,
                out_channels,
                act=act,
                alpha=False, )
        else:
            self.conv2 = Conv(
                hidden_channels,
                out_channels,
                ksize=3,
                stride=1,
                bias=bias,
                act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Layer):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 bias=False,
                 act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.LayerList([
            nn.MaxPool2D(
                kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [mp(x) for mp in self.maxpoolings], axis=1)
        x = self.conv2(x)
        return x


class SPPFLayer(nn.Layer):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=5,
                 bias=False,
                 act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2D(
            kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = paddle.concat([x, y1, y2, y3], axis=1)
        out = self.conv2(concats)
        return out


class ParalleNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=1.0,
                 depthwise=False,
                 num_blocks=3,
                 bias=False,
                 act="silu",
                 use_repconv=False):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1s = nn.LayerList([
            BaseConv(
                in_channels,
                hidden_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])

        if use_repconv:
            self.conv2s = nn.LayerList([
                RepVggBlock(
                    hidden_channels,
                    out_channels,
                    act=act,
                    alpha=False, ) for _ in range(num_blocks)
            ])
        else:
            self.conv2s = nn.LayerList([
                BaseConv(
                    hidden_channels,
                    out_channels,
                    ksize=3,
                    stride=1,
                    bias=bias,
                    act=act) for _ in range(num_blocks)
            ])

        self.conv3 = BaseConv(
            out_channels * num_blocks,
            hidden_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        feats = [m(x) for m in self.conv1s]
        feats = [m(x) for m in self.conv2s]

        y = paddle.concat(feats, axis=1)
        y = self.conv3(y)

        if self.add_shortcut:
            y = y + x

        return y


class RepConvNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=1.0,
                 depthwise=False,
                 num_blocks=3,
                 bias=False,
                 act="silu",
                 use_repconv=False,
                 rep_fmt='origin'):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # self.conv1s = nn.LayerList([
        #     BaseConv(
        #         in_channels,
        #         hidden_channels,
        #         ksize=1,
        #         stride=1,
        #         bias=bias,
        #         act=act) for _ in range(num_blocks)
        # ])
        self.rep_fmt = rep_fmt
        self.conv1s = nn.LayerList([nn.Identity() for _ in range(num_blocks)])

        self.conv2s = nn.LayerList([
            RepVggBlock(
                hidden_channels,
                hidden_channels,
                act=act,
                alpha=False, ) for _ in range(num_blocks)
        ])

        self.conv3 = nn.Identity()

        # self.conv3 = BaseConv(
        #     hidden_channels,
        #     out_channels,
        #     ksize=1,
        #     stride=1,
        #     bias=bias,
        #     act=act)

        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        _y = x
        outputs = []
        for m1, m2 in zip(self.conv1s, self.conv2s):
            y = m2(m1(_y))

            if self.add_shortcut:
                _y = _y + y
            else:
                _y = y

            outputs.append(_y)

        y = self.conv3(_y)

        if self.rep_fmt == 'add':
            return sum(outputs)
        else:
            return y
        # if self.add_shortcut:
        #     y = y + x
        # return y


class CSPLayer(nn.Layer):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu",
                 use_repconv=False,
                 block_fmt='bottle',
                 csp_fmt='origin',
                 rep_fmt='origin'):
        super(CSPLayer, self).__init__()

        hidden_channels = int(out_channels * expansion)

        self.csp_fmt = csp_fmt
        if csp_fmt == 'origin':
            self.conv1 = BaseConv(
                in_channels,
                hidden_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
            self.conv2 = BaseConv(
                in_channels,
                hidden_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
            self.conv3 = BaseConv(
                hidden_channels * 2,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)

        if csp_fmt == 'add':
            self.conv1 = BaseConv(
                in_channels,
                hidden_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
            self.conv2 = BaseConv(
                in_channels,
                hidden_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
            # self.conv1 = nn.Identity()
            # self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

        # self.conv1 = BaseConv(
        #     in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        # self.conv2 = BaseConv(
        #     in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)

        if block_fmt == 'bottle':
            self.bottlenecks = nn.Sequential(*[
                BottleNeck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                    depthwise=depthwise,
                    bias=bias,
                    act=act,
                    use_repconv=use_repconv) for _ in range(num_blocks)
            ])
        elif block_fmt == 'parallel':
            self.bottlenecks = ParalleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                num_blocks=num_blocks,
                bias=bias,
                act=act,
                use_repconv=use_repconv, )

        elif block_fmt == 'rep':
            self.bottlenecks = RepConvNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                num_blocks=num_blocks,
                bias=bias,
                act=act,
                use_repconv=use_repconv,
                rep_fmt=rep_fmt)

    def forward(self, x):
        if self.csp_fmt == 'origin':
            x_1 = self.conv1(x)
            x_1 = self.bottlenecks(x_1)
            x_2 = self.conv2(x)
            x = paddle.concat([x_1, x_2], axis=1)
            x = self.conv3(x)
            return x
        elif self.csp_fmt == 'add':
            x_1 = self.conv1(x)
            x_1 = self.bottlenecks(x_1)
            x_2 = self.conv2(x)
            x = x_1 + x_2
            x = self.conv3(x)
            return x


@register
@serializable
class CSPDarkNet(nn.Layer):
    """
    CSPDarkNet backbone.
    Args:
        arch (str): Architecture of CSPDarkNet, from {P5, P6, X}, default as X,
            and 'X' means used in YOLOX, 'P5/P6' means used in YOLOv5.
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    # 'X' means setting used in YOLOX, 'P5/P6' means setting used in YOLOv5.
    arch_settings = {
        'X': [[64, 128, 3, True, False], [128, 256, 9, True, False],
              [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='X',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(CSPDarkNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv
        arch_setting = self.arch_settings[arch]
        base_channels = int(arch_setting[0][0] * width_mult)

        # Note: differences between the latest YOLOv5 and the original YOLOX
        # 1. self.stem, use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 2. use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 3. put SPPF before(YOLOv5) or SPP after(YOLOX) the last cspdark block's CSPLayer
        # 4. whether SPPF(SPP)'CSPLayer add shortcut, True in YOLOv5, False in YOLOX
        if arch in ['P5', 'P6']:
            # in the latest YOLOv5, use Conv stem, and SPPF (fast, only single spp kernal size)
            self.stem = Conv(
                3, base_channels, ksize=6, stride=2, bias=False, act=act)
            spp_kernal_sizes = 5
        elif arch in ['X']:
            # in the original YOLOX, use Focus stem, and SPP (three spp kernal sizes)
            self.stem = Focus(
                3, base_channels, ksize=3, stride=1, bias=False, act=act)
            spp_kernal_sizes = (5, 9, 13)
        else:
            raise AttributeError("Unsupported arch type: {}".format(arch))

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(
                    in_channels, out_channels, 3, 2, bias=False, act=act))
            stage.append(conv_layer)
            layers_num += 1

            if use_spp and arch in ['X']:
                # in YOLOX use SPPLayer
                spp_layer = self.add_sublayer(
                    'layers{}.stage{}.spp_layer'.format(layers_num, i + 1),
                    SPPLayer(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernal_sizes,
                        bias=False,
                        act=act))
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = self.add_sublayer(
                'layers{}.stage{}.csp_layer'.format(layers_num, i + 1),
                CSPLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    shortcut=shortcut,
                    depthwise=depthwise,
                    bias=False,
                    act=act))
            stage.append(csp_layer)
            layers_num += 1

            if use_spp and arch in ['P5', 'P6']:
                # in latest YOLOv5 use SPPFLayer instead of SPPLayer
                sppf_layer = self.add_sublayer(
                    'layers{}.stage{}.sppf_layer'.format(layers_num, i + 1),
                    SPPFLayer(
                        out_channels,
                        out_channels,
                        ksize=5,
                        bias=False,
                        act=act))
                stage.append(sppf_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]
