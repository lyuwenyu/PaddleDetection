import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_, xavier_uniform_
from ppdet.modeling.shape_spec import ShapeSpec

import math

# TransformerDecoder
# TransformerEncoderLayer


@register
@serializable
class TFPN(nn.Layer):
    __shared__ = ['act', ]

    def __init__(self,
                 in_channels,
                 hidden_dim=256,
                 num_layers=6,
                 nhead=8,
                 position_embed_type='sine',
                 dim_feedforward=1024,
                 dropout=0.1,
                 add_position_perlayer=False,
                 act='relu'):
        super().__init__()

        # in_channels = 512
        # hidden_dim = 256
        # num_layers = 6
        # position_embed_type = 'sine'
        # activation = 'relu'
        # dim_feedforward = 1024
        # nhead = 8
        # dropout = 0.1

        # assert len(in_channels) == 1, ''
        # in_channels = in_channels[0]

        self.input_projects = nn.LayerList(
            [nn.Conv2D(
                c, hidden_dim, kernel_size=1) for c in in_channels])

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation=act)
        # encoder = nn.TransformerEncoder(encoder_layer, num_layers, )
        self.encoders = nn.LayerList([
            nn.TransformerEncoder(
                encoder_layer,
                num_layers, ) for _ in in_channels
        ])

        self._out_channels = [hidden_dim for _ in in_channels]
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # conv_init_(self.input_projects)

    def forward(self, feats, for_mot=False):

        # feats = [m(x) for m, x in zip(self.input_projects, feats)]
        # N, D, H, W = src_proj.shape

        # src_mask = paddle.ones([N, H, W], dtype='bool')
        # pos_embed = self.position_embedding(src_mask)

        # src_proj = src_proj + pos_embed
        outputs = []
        memory = None

        for i, feat in enumerate(feats):

            feat = self.input_projects[i](feat)
            N, D, H, W = feat.shape

            feat = feat + F.interpolate(
                memory, size=(H, W)) if memory is not None else feat

            # src_mask = paddle.ones([N, H, W], dtype='bool')
            # pos_embed = self.position_embedding(src_mask)
            # src_proj = src_proj + pos_embed

            src_flatten = feat.flatten(2).transpose([0, 2, 1])

            src_mask = None
            memory = self.encoders[i](src_flatten, src_mask)  # N (HW) D
            memory = memory.transpose([0, 2, 1]).reshape([N, D, H, W])
            outputs.append(memory)

        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
