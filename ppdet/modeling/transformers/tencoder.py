import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_, xavier_uniform_
from ppdet.modeling.shape_spec import ShapeSpec

import math

# TransformerDecoder
# TransformerEncoderLayer

from .tencoder_utils import TransformerEncoder as PPTransformerEncoder


@register
@serializable
class TEncoder(nn.Layer):
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
                 skip_connection=False,
                 fused_multi_stages=False,
                 return_intermediate=False,
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

        self.skip_connection = skip_connection
        self.fused_multi_stages = skip_connection and fused_multi_stages
        self.return_intermediate = return_intermediate

        if not skip_connection:
            assert len(in_channels) == 1, ''
            self.input_projects = nn.Sequential(
                nn.Conv2D(
                    in_channels[-1], hidden_dim, kernel_size=1))

        else:
            assert len(in_channels) == 3, ''
            self.input_projects = nn.LayerList(
                [nn.Conv2D(
                    c, hidden_dim, kernel_size=1) for c in in_channels])

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation=act)
        self.encoder = PPTransformerEncoder(
            encoder_layer, num_layers, return_intermediate=return_intermediate)

        self.fpns = nn.LayerList([
            nn.Sequential(
                nn.Conv2DTranspose(
                    hidden_dim, hidden_dim, 2, stride=2)), Identity(),
            nn.Sequential(nn.MaxPool2D(2, 2))
        ])

        self._out_channels = [hidden_dim, hidden_dim, hidden_dim]
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        for m in self.input_projects:
            conv_init_(m)

    def forward(self, feats, for_mot=False):

        if not self.skip_connection:
            src_proj = self.input_projects(feats[-1])
        else:
            feats = [m(x) for m, x in zip(self.input_projects, feats)]

            if self.fused_multi_stages:
                _, _, h, w = feats[1].shape
                src_proj = feats[1] + F.interpolate(
                    feats[0], scale_factor=0.5) + F.interpolate(
                        feats[-1], scale_factor=2.0)
                # src_proj = feats[1] + F.interpolate(feats[0], (h, w)) + F.interpolate(feats[-1], (h, w))
            else:
                src_proj = feats[1]

        N, D, H, W = src_proj.shape

        src_mask = paddle.ones([N, H, W], dtype='bool')
        pos_embed = self.position_embedding(src_mask)
        src_proj = src_proj + pos_embed

        src_flatten = src_proj.flatten(2).transpose([0, 2, 1])

        src_mask = None
        memory = self.encoder(src_flatten, src_mask)  # N (HW) D

        if not self.return_intermediate:
            memory = memory.transpose([0, 2, 1]).reshape([N, D, H, W])
            outputs = [m(memory) for m in self.fpns]
            if self.skip_connection:
                outputs = [x + y for x, y in zip(feats, outputs)]

        else:
            outputs = {}
            for i, mem in enumerate(memory):
                mem = mem.transpose([0, 2, 1]).reshape([N, D, H, W])
                _outputs = [m(mem) for m in self.fpns]
                if self.skip_connection:
                    _outputs = [x + y for x, y in zip(feats, _outputs)]
                name = str(i) if i < len(memory) - 1 else 'last'
                outputs[name] = _outputs

        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]


class PositionEmbedding(nn.Layer):
    def __init__(self,
                 num_pos_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=None,
                 embed_type='sine',
                 num_embeddings=50,
                 offset=0.):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        self.offset = offset
        self.eps = 1e-6
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"not supported {self.embed_type}")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, C, H, W]
        """
        assert mask.dtype == paddle.bool
        if self.embed_type == 'sine':
            mask = mask.astype('float32')
            y_embed = mask.cumsum(1, dtype='float32')
            x_embed = mask.cumsum(2, dtype='float32')
            if self.normalize:
                y_embed = (y_embed + self.offset) / (
                    y_embed[:, -1:, :] + self.eps) * self.scale
                x_embed = (x_embed + self.offset) / (
                    x_embed[:, :, -1:] + self.eps) * self.scale

            dim_t = 2 * (paddle.arange(self.num_pos_feats) //
                         2).astype('float32')
            dim_t = self.temperature**(dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t
            pos_x = paddle.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            pos_y = paddle.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                axis=4).flatten(3)
            pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
            return pos

        elif self.embed_type == 'learned':
            h, w = mask.shape[-2:]
            i = paddle.arange(w)
            j = paddle.arange(h)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            pos = paddle.concat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                axis=-1).transpose([2, 0, 1]).unsqueeze(0).tile(mask.shape[0],
                                                                1, 1, 1)
            return pos
        else:
            raise ValueError(f"not supported {self.embed_type}")


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


## 

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .utils import _get_clones
from .position_encoding import PositionEmbedding
from ..layers import MultiHeadAttention, _convert_attention_mask
from ..initializer import linear_init_, conv_init_, xavier_uniform_


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        # q = k = self.with_pos_embed(src, pos_embed)
        q = k = src if pos_embed is None else src + pos_embed

        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
@serializable
class TEncoderWithPos(nn.Layer):
    __shared__ = ['act', ]

    def __init__(self,
                 in_channels,
                 hidden_dim=256,
                 num_layers=6,
                 nhead=8,
                 position_embed_type='sine',
                 dim_feedforward=1024,
                 dropout=0.1,
                 act='relu'):
        super().__init__()

        assert len(in_channels) == 1, ''
        in_channels = in_channels[0]

        self.input_project = nn.Conv2D(in_channels, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation=act)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers, )

        # encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward,)
        # self.encoder = TransformerEncoder()

        self.fpns = nn.LayerList([
            nn.Sequential(
                nn.Conv2DTranspose(
                    hidden_dim, hidden_dim, 2, stride=2)), Identity(),
            nn.Sequential(nn.MaxPool2D(2, 2))
        ])

        self._out_channels = [hidden_dim, hidden_dim, hidden_dim]

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        conv_init_(self.input_project)

    def forward(self, feats, for_mot=False):

        src_proj = self.input_project(feats[-1])
        N, D, H, W = src_proj.shape

        src_mask = paddle.ones([N, H, W], dtype='bool')
        # pos_embed = self.position_embedding(src_mask)

        pos_embed = self.position_embedding(src_mask).flatten(2).transpose(
            [0, 2, 1])
        src_flatten = src_proj.flatten(2).transpose([0, 2, 1])

        src_mask = _convert_attention_mask(src_mask, src_flatten.dtype)
        # print(src_mask.shape)
        # src_mask = src_mask.reshape([N, 1, 1, -1])
        src_mask = src_mask.flatten(1).unsqueeze(1).unsqueeze(1)

        memory = self.encoder(
            src_flatten, src_mask=src_mask, pos_embed=pos_embed)
        memory = memory.transpose([0, 2, 1]).reshape([N, D, H, W])

        # src_mask = None
        # memory = self.encoder(src_flatten, src_mask)  # N (HW) D
        # memory = memory.transpose([0, 2, 1]).reshape([N, D, H, W])

        outputs = [m(memory) for m in self.fpns]

        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]

        # src_proj = self.input_proj(src[-1])
        # bs, c, h, w = src_proj.shape
        # # flatten [B, C, H, W] to [B, HxW, C]
        # src_flatten = src_proj.flatten(2).transpose([0, 2, 1])
        # if src_mask is not None:
        #     src_mask = F.interpolate(
        #         src_mask.unsqueeze(0).astype(src_flatten.dtype),
        #         size=(h, w))[0].astype('bool')
        # else:
        #     src_mask = paddle.ones([bs, h, w], dtype='bool')
        # pos_embed = self.position_embedding(src_mask).flatten(2).transpose(
        #     [0, 2, 1])

        # src_mask = _convert_attention_mask(src_mask, src_flatten.dtype)
        # src_mask = src_mask.reshape([bs, 1, 1, -1])

        # memory = self.encoder(
        #     src_flatten, src_mask=src_mask, pos_embed=pos_embed)


import numpy as np
from ppdet.modeling.backbones.vision_transformer import Block


@register
@serializable
class ViTEncoder(nn.Layer):
    __shared__ = ['act', ]

    def __init__(self,
                 in_channels,
                 hidden_dim=256,
                 num_layers=6,
                 nhead=8,
                 position_embed_type='sine',
                 dim_feedforward=1024,
                 dropout=0.1,
                 act='relu',
                 use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        assert len(in_channels) == 1, ''
        in_channels = in_channels[0]

        epsilon = 1e-5
        drop_path_rate = 0.
        drop_rate = 0.
        attn_drop_rate = 0.
        mlp_ratio = dim_feedforward // hidden_dim
        norm_layer = 'nn.LayerNorm'
        init_values = 0.1
        qkv_bias = True
        qk_scale = None
        epsilon = 1e-6

        #   mlp_ratio: 4
        #   qkv_bias: True
        #   drop_rate: 0.0
        #   drop_path_rate: 0.2
        #   init_values: 0.1
        #   final_norm: False
        #   use_rel_pos_bias: False
        #   use_sincos_pos_emb: True

        if act == 'relu':
            act_layer = nn.ReLU
        elif act == 'gelu':
            act_layer = nn.GELU
        else:
            raise RuntimeError('')

        dpr = np.linspace(0, drop_path_rate, num_layers)
        self.blocks = nn.LayerList([
            Block(
                dim=hidden_dim,
                num_heads=nhead,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                act_layer=act_layer,
                window_size=None,
                epsilon=epsilon) for i in range(num_layers)
        ])

        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type)

        self.input_project = nn.Conv2D(in_channels, hidden_dim, kernel_size=1)
        self.fpns = nn.LayerList([
            nn.Sequential(
                nn.Conv2DTranspose(
                    hidden_dim, hidden_dim, 2, stride=2)), Identity(),
            nn.Sequential(nn.MaxPool2D(2, 2))
        ])

        self._out_channels = [hidden_dim, hidden_dim, hidden_dim]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]

    def forward(self, feats, for_mot=False):

        src_proj = self.input_project(feats[-1])
        N, D, H, W = src_proj.shape

        src_mask = paddle.ones([N, H, W], dtype='bool')
        pos_embed = self.position_embedding(src_mask).flatten(2).transpose(
            [0, 2, 1])
        src_flatten = src_proj.flatten(2).transpose([0, 2, 1]) + pos_embed

        x = src_flatten

        feats = []
        rel_pos_bias = None
        for _, blk in enumerate(self.blocks):
            if self.use_checkpoint and not self.training:
                x = paddle.distributed.fleet.utils.recompute(
                    blk, x, rel_pos_bias, **{"preserve_rng_state": True})
            else:
                x = blk(x, rel_pos_bias)

            xp = x.transpose([0, 2, 1]).reshape([N, D, H, W])

            feats.append(xp)

        outputs = [m(feats[-1]) for m in self.fpns]

        return outputs
