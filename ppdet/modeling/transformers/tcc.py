import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_, xavier_uniform_
from ppdet.modeling.shape_spec import ShapeSpec

import math


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Interpolate(nn.Layer):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        # _, _, h, w = x.shape
        # size = [int(h * self.scale_factor), int(w * self.scale_factor)]
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


@register
@serializable
class TCC(nn.Layer):
    __shared__ = ['act', ]

    def __init__(self, in_channels, act='relu'):
        super().__init__()

        assert len(in_channels) == 3, ''

        self.projects = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(in_channels[0], in_channels[1], 1, 1),
                Interpolate(0.5)),
            Identity(),
            nn.Sequential(
                nn.Conv2D(in_channels[2], in_channels[1], 1, 1),
                Interpolate(2.0)),
        ])
        self.local_context = nn.Conv2D(
            in_channels[1], 128, kernel_size=3, dilation=2, padding=2)

        self.K = 3
        self.global_mask = nn.Sequential(
            nn.Conv2D(in_channels[1], self.K, 1, 1),
            nn.AdaptiveMaxPool2D(
                1, return_mask=True))

        self.cross_attn = nn.MultiHeadAttention(
            in_channels[1],
            8,
            0.1, )
        # layer = nn.TransformerDecoderLayer(in_channels[1], 8, 1024, 0, 'relu')
        # self.decoder = nn.TransformerDecoder(layer, num_layers=3)

        self._out_channels = [in_channels[1]
                              for _ in range(len(in_channels))][:1]

    def forward(self, feats, for_mot=False):

        feat = sum([m(x) for m, x in zip(self.projects, feats)])
        N, C, H, W = feat.shape

        query = self.local_context(feat).flatten(2).transpose(
            [0, 2, 1])  # n hw c

        # F.adaptive_max_pool2d()

        global_val, global_msk = self.global_mask(feat)
        global_score = F.max_unpool2d(
            global_val, global_msk, kernel_size=feat.shape[-2:])
        ni, _, hi, wi = global_score.nonzero(as_tuple=True)
        global_feats = feat.transpose([0, 2, 3, 1])[ni, hi, wi].reshape(
            [N, self.K, C])  # n l c
        key = value = paddle.concat([query, global_feats], axis=1)

        attn_mask = None
        feat = self.cross_attn(query, key, value, attn_mask)

        feat = feat.transpose([0, 2, 1]).reshape([N, C, H, W])

        return [feat]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
