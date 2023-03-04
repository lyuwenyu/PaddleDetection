# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from numpy import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from ..heads.detr_head import MLP
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_,
                           bias_init_with_prob)
from .utils import (_get_clones, deformable_attention_core_func,
                    get_valid_ratio, get_contrastive_denoising_training_group,
                    get_sine_pos_embed, inverse_sigmoid)

__all__ = ['DINOTransformer']


class MSDeformableAttention(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=1.0):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile(
            [1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(
            1, self.num_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling
        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            offset_normalizer = value_spatial_shapes.flip([1]).reshape(
                [1, 1, 1, self.num_levels, 1, 2])
            sampling_locations = reference_points.reshape([
                bs, Len_q, 1, self.num_levels, 1, 2
            ]) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] *
                0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = deformable_attention_core_func(
            value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class DINOTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                               n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes,
                src_mask=None,
                query_pos_embed=None):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(src, query_pos_embed),
            reference_points=reference_points,
            value=src,
            value_spatial_shapes=spatial_shapes,
            value_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src


class DINOTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(DINOTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        # for i, (H, W) in enumerate(spatial_shapes.tolist()):
        for i, (H, W) in enumerate(spatial_shapes):

            # ref_y, ref_x = paddle.meshgrid(
            #     paddle.linspace(0.5, H - 0.5, H),
            #     paddle.linspace(0.5, W - 0.5, W))
            ref_y, ref_x = paddle.meshgrid(paddle.arange(H), paddle.arange(W))
            ref_y = ref_y + 0.5
            ref_x = ref_x + 0.5

            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] *
                                                    H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] *
                                                    W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self,
                feat,
                spatial_shapes,
                feat_mask=None,
                query_pos_embed=None,
                valid_ratios=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [feat.shape[0], spatial_shapes.shape[0], 2])
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for layer in self.layers:
            feat = layer(feat, reference_points, spatial_shapes, feat_mask,
                         query_pos_embed)

        return feat


class DINOTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr,
                                 bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr,
                                 bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = attn_mask.astype('bool')
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


from collections import defaultdict


class DINOTransformerDecoder(nn.Layer):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 return_intermediate=True,
                 path_type='base',
                 drop_p=0.2,
                 look_forward_twice=True,
                 sqr_epoch=100000,
                 use_sin_query_pos_embed=True,
                 sin_query_pos_ratio=2,
                 learn_sin_query_pos_embed=False,
                 set_query_pos_embed_none=False):
        super(DINOTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.use_sin_query_pos_embed = use_sin_query_pos_embed
        self.sin_query_pos_ratio = sin_query_pos_ratio

        self.path_type = path_type
        self.look_forward_twice = look_forward_twice
        self.drop_p = drop_p
        self.sqr_epoch = sqr_epoch
        self.learn_sin_query_pos_embed = learn_sin_query_pos_embed
        self.set_query_pos_embed_none = set_query_pos_embed_none

        assert path_type in ('base', 'drop_v1', 'drop_v2', 'drop_v2', 'sqr'), ''

        self.norm = nn.LayerNorm(
            hidden_dim,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                valid_ratios=None,
                attn_mask=None,
                memory_mask=None,
                epoch=-1):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [memory.shape[0], memory_spatial_shapes.shape[0], 2])

        output = tgt
        intermediate = []
        dec_out_bboxes = []
        dec_out_logits = []

        drop_i = random.randint(1, self.num_layers - 1)
        drop_d = random.randint(3, self.num_layers)
        dynamic_k = 0

        reference_points_input_list = [reference_points, ]

        dec_query_set = defaultdict(list)
        dec_query_set[0].append(('0', reference_points, reference_points, None))
        ks = [1, 2, 3, 5, 8, 13, 21]
        dec_out_bboxes_list = []
        dec_out_logits_list = []

        for i, layer in enumerate(self.layers):

            if self.path_type == 'sqr' and epoch < self.sqr_epoch:
                for j, (_name, _iboxes, _, _) in enumerate(dec_query_set[i]):
                    reference_points = _iboxes

                    reference_points_input = reference_points.detach(
                    ).unsqueeze(2) * valid_ratios.tile([1, 1, 2]).unsqueeze(1)

                    if self.use_sin_query_pos_embed:
                        if self.sin_query_pos_ratio == 2:
                            query_pos_embed = get_sine_pos_embed(
                                reference_points_input[..., 0, :],
                                self.hidden_dim // 2)
                        elif self.sin_query_pos_ratio == 4:
                            query_pos_embed = get_sine_pos_embed(
                                reference_points_input[..., 0, :],
                                self.hidden_dim // 4)
                    else:
                        query_pos_embed = reference_points.detach()

                    query_pos_embed = query_pos_head(query_pos_embed)

                    output = layer(output, reference_points_input, memory,
                                   memory_spatial_shapes, attn_mask,
                                   memory_mask, query_pos_embed)

                    inter_ref_points = F.sigmoid(bbox_head[i](
                        output) + inverse_sigmoid(reference_points.detach()))

                    dec_logit = score_head[i](output)

                    if self.return_intermediate:
                        intermediate.append(self.norm(output))

                        if self.look_forward_twice:
                            if i == 0:
                                dec_out_bboxes.append(inter_ref_points)
                            else:
                                dec_out_bboxes.append(
                                    F.sigmoid(bbox_head[i](output) +
                                              inverse_sigmoid(
                                                  reference_points)))
                        else:
                            dec_out_bboxes.append(inter_ref_points)

                        dec_out_logits.append(dec_logit)

                        dec_query_set[i + 1].append(
                            ('{}{}'.format(_name, i + 1), inter_ref_points,
                             dec_out_bboxes[-1], dec_out_logits[-1]))

                for _, (_n, _, _boxes,
                        _logits) in enumerate(dec_query_set[i + 1][::-1]):
                    dec_out_bboxes_list.append(_boxes)
                    dec_out_logits_list.append(_logits)
                    # print(i + 1, _n)

                _k = ks[i + 1] - len(dec_query_set[i + 1])
                dec_query_set[i + 1].extend(dec_query_set[i][:_k])

            else:
                # drop block
                if self.training and self.path_type == 'drop_v1' and i == drop_i and random.uniform(
                        0., 1.) < self.drop_p:
                    continue

                # drop path
                elif self.training and self.path_type == 'drop_v2' and i == drop_i and random.uniform(
                        0., 1.) < self.drop_p:
                    # reference_points_input_list = reference_points_input_list[:-1]
                    reference_points = reference_points_input_list[-2]

                # dynamic layers
                elif self.training and self.path_type == 'drop_v3':
                    if i > drop_d:
                        continue

                elif self.training and self.path_type == 'dynamic':
                    _k = random.randint(dynamic_k,
                                        len(reference_points_input_list))
                    reference_points = reference_points_input_list[_k]
                    dynamic_k = _k

                else:
                    pass

                reference_points_input = reference_points.detach().unsqueeze(
                    2) * valid_ratios.tile([1, 1, 2]).unsqueeze(1)

                if self.set_query_pos_embed_none:
                    query_pos_embed = None

                elif not self.learn_sin_query_pos_embed:
                    if self.use_sin_query_pos_embed:
                        if self.sin_query_pos_ratio == 2:
                            query_pos_embed = get_sine_pos_embed(
                                reference_points_input[..., 0, :],
                                self.hidden_dim // 2)
                        elif self.sin_query_pos_ratio == 4:
                            query_pos_embed = get_sine_pos_embed(
                                reference_points_input[..., 0, :],
                                self.hidden_dim // 4)
                    else:
                        query_pos_embed = reference_points.detach()
                        # query_pos_embed = inverse_sigmoid(reference_points.detach())

                    query_pos_embed = query_pos_head(query_pos_embed)

                else:
                    query_pos_embed = query_pos_head.weight  # .weight.unsqueeze(0).tile([bs, 1, 1])

                    # t = paddle.zeros_like(output)
                    # t[:, -300:, :] = query_pos_embed.unsqueeze(0).tile([t.shape[0], 1, 1])

                    t = paddle.zeros([
                        output.shape[1] - query_pos_embed.shape[0],
                        query_pos_embed.shape[-1]
                    ])
                    t = paddle.concat([t, query_pos_embed], axis=0)

                    query_pos_embed = t

                output = layer(output, reference_points_input, memory,
                               memory_spatial_shapes, attn_mask, memory_mask,
                               query_pos_embed)

                if not self.training and self.path_type == 'drop_v1' and i in list(
                        range(1, self.num_layers - 1)):
                    output *= 1. / (1 - 1. /
                                    (self.num_layers - 2) * self.drop_p)

                inter_ref_points = F.sigmoid(bbox_head[i](
                    output) + inverse_sigmoid(reference_points.detach()))
                dec_logit = score_head[i](output)

                if self.return_intermediate:
                    intermediate.append(self.norm(output))
                    if i == 0:
                        dec_out_bboxes.append(inter_ref_points)
                    else:
                        dec_out_bboxes.append(
                            F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                                reference_points)))
                    dec_out_logits.append(dec_logit)

                reference_points = inter_ref_points

                reference_points_input_list.append(reference_points)

        if self.return_intermediate and self.path_type == 'sqr' and epoch < self.sqr_epoch:
            return None, paddle.stack(dec_out_bboxes_list), paddle.stack(
                dec_out_logits_list)

        elif self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                dec_out_bboxes), paddle.stack(dec_out_logits)

        return output, reference_points, dec_logit


@register
class DINOTransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=900,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_feat_channels=[512, 1024, 2048],
                 num_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 eps=1e-2,
                 path_type='base',
                 drop_p=0.2,
                 dn_epoch=10000000,
                 mlp_activation='relu',
                 num_bbox_head_layers=3,
                 num_query_pos_head_layers=2,
                 keep_mlp_bias_weight_decay=True,
                 sqr_epoch=1000000,
                 use_sin_query_pos_embed=True,
                 sin_query_pos_ratio=2,
                 topk_sorted=True,
                 learn_sin_query_pos_embed=False,
                 set_query_pos_embed_none=False,
                 add_lvl_pos_embed_flatten=False):
        super(DINOTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.dn_epoch = dn_epoch
        self.sqr_epoch = sqr_epoch
        self.topk_sorted = topk_sorted
        self.add_lvl_pos_embed_flatten = add_lvl_pos_embed_flatten
        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            encoder_layer = DINOTransformerEncoderLayer(
                hidden_dim, nhead, dim_feedforward, dropout, activation,
                num_levels, num_encoder_points)
            self.encoder = DINOTransformerEncoder(encoder_layer,
                                                  num_encoder_layers)

        decoder_layer = DINOTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = DINOTransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            path_type=path_type,
            drop_p=drop_p,
            sqr_epoch=sqr_epoch,
            use_sin_query_pos_embed=use_sin_query_pos_embed,
            learn_sin_query_pos_embed=learn_sin_query_pos_embed,
            set_query_pos_embed_none=set_query_pos_embed_none)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes + 1, hidden_dim, padding_idx=num_classes)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # position embedding
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=-0.5)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)
        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.use_sin_query_pos_embed = use_sin_query_pos_embed
        self.sin_query_pos_ratio = sin_query_pos_ratio

        self.learn_sin_query_pos_embed = learn_sin_query_pos_embed
        if not learn_sin_query_pos_embed:
            if use_sin_query_pos_embed:
                if sin_query_pos_ratio == 2:
                    self.query_pos_head = MLP(
                        2 * hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        num_layers=num_query_pos_head_layers,
                        activation=mlp_activation,
                        keep_bias_weight_decay=keep_mlp_bias_weight_decay)
                elif sin_query_pos_ratio == 4:
                    self.query_pos_head = MLP(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        num_layers=num_query_pos_head_layers,
                        activation=mlp_activation,
                        keep_bias_weight_decay=keep_mlp_bias_weight_decay)
            else:
                self.query_pos_head = MLP(
                    4,
                    hidden_dim,
                    hidden_dim,
                    num_layers=num_query_pos_head_layers,
                    activation=mlp_activation,
                    keep_bias_weight_decay=keep_mlp_bias_weight_decay)

        else:
            self.query_pos_head = nn.Embedding(num_queries, hidden_dim)

        self.set_query_pos_embed_none = set_query_pos_embed_none
        if set_query_pos_embed_none:
            self.query_pos_head = None

        # encoder head
        self.enc_output = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
            if keep_mlp_bias_weight_decay else nn.Linear(
                hidden_dim,
                hidden_dim,
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        # self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_score_head = nn.Linear(
            hidden_dim,
            num_classes) if keep_mlp_bias_weight_decay else nn.Linear(
                hidden_dim,
                num_classes,
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        self.enc_bbox_head = MLP(
            hidden_dim,
            hidden_dim,
            4,
            num_layers=num_bbox_head_layers,
            activation=mlp_activation,
            keep_bias_weight_decay=keep_mlp_bias_weight_decay)
        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            if keep_mlp_bias_weight_decay else nn.Linear(
                hidden_dim,
                num_classes,
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim,
                hidden_dim,
                4,
                num_layers=num_bbox_head_layers,
                activation=mlp_activation,
                keep_bias_weight_decay=keep_mlp_bias_weight_decay)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        normal_(self.level_embed.weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)

        if not self.learn_sin_query_pos_embed:
            xavier_uniform_(self.query_pos_head.layers[0].weight)
            xavier_uniform_(self.query_pos_head.layers[1].weight)
        else:
            xavier_uniform_(self.query_pos_head.weight)

        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape], }

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)), ('norm', nn.BatchNorm2D(
                            self.hidden_dim,
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats, pad_mask=None):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for i, feat in enumerate(proj_feats):
            bs, _, h, w = paddle.shape(feat)
            spatial_shapes.append(paddle.concat([h, w]))
            # [b,c,h,w] -> [b,h*w,c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            if pad_mask is not None:
                mask = F.interpolate(pad_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            # [b, h*w, c]
            pos_embed = self.position_embedding(mask)
            lvl_pos_embed = pos_embed + self.level_embed.weight[i]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            if pad_mask is not None:
                # [b, h*w]
                mask_flatten.append(mask.flatten(1))

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        # [b, l]
        mask_flatten = None if pad_mask is None else paddle.concat(mask_flatten,
                                                                   1)
        # [b, l, c]
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [num_levels, 2]
        spatial_shapes = paddle.stack(spatial_shapes)
        # [b, num_levels, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)
        return (feat_flatten, spatial_shapes, mask_flatten,
                lvl_pos_embed_flatten, valid_ratios)

    def forward(self, feats, pad_mask=None, gt_meta=None):
        # input projection and embedding
        (feat_flatten, spatial_shapes, mask_flatten, lvl_pos_embed_flatten,
         valid_ratios) = self._get_encoder_input(feats, pad_mask)

        # encoder
        if self.num_encoder_layers > 0:
            memory = self.encoder(feat_flatten, spatial_shapes, mask_flatten,
                                  lvl_pos_embed_flatten, valid_ratios)
        else:
            memory = feat_flatten + lvl_pos_embed_flatten if self.add_lvl_pos_embed_flatten else feat_flatten

        # solve hang during distributed training
        memory = memory + self.denoising_class_embed.weight.sum() * 0.

        # prepare denoising training
        if self.training and gt_meta['epoch_id'] < self.dn_epoch:
            denoising_class, denoising_bbox, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
            memory, spatial_shapes, mask_flatten, denoising_class,
            denoising_bbox)

        # decoder
        _, dec_out_bboxes, dec_out_logits = self.decoder(
            target,
            init_ref_points,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            valid_ratios,
            attn_mask,
            mask_flatten,
            epoch=gt_meta.get('epoch_id', -1))

        return (dec_out_bboxes, dec_out_logits, enc_topk_bboxes,
                enc_topk_logits, dn_meta)

    def _get_encoder_output_anchors(self,
                                    memory,
                                    spatial_shapes,
                                    memory_mask=None,
                                    grid_size=0.05):
        output_anchors = []
        idx = 0
        for lvl, (h, w) in enumerate(spatial_shapes):
            if memory_mask is not None:
                mask_ = memory_mask[:, idx:idx + h * w].reshape([-1, h, w])
                valid_H = paddle.sum(mask_[:, :, 0], 1)
                valid_W = paddle.sum(mask_[:, 0, :], 1)
            else:
                valid_H, valid_W = h, w

            grid_y, grid_x = paddle.meshgrid(
                paddle.linspace(0, h - 1, h), paddle.linspace(0, w - 1, w))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.stack([valid_W, valid_H], -1).reshape(
                [-1, 1, 1, 2]).astype(grid_xy.dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            output_anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))
            idx += h * w

        output_anchors = paddle.concat(output_anchors, 1)
        # valid_mask = ((output_anchors > self.eps) &
        #               (output_anchors < 1 - self.eps)).all(-1, keepdim=True)
        valid_mask = ((output_anchors > self.eps) *
                      (output_anchors < 1 - self.eps)).all(-1, keepdim=True)

        output_anchors = paddle.log(output_anchors / (1 - output_anchors))
        # if memory_mask is not None:
        #     valid_mask = (valid_mask & (memory_mask.unsqueeze(-1) > 0)) > 0
        if memory_mask is not None:
            valid_mask = (valid_mask * (memory_mask.unsqueeze(-1) > 0)) > 0

        output_anchors = paddle.where(valid_mask, output_anchors,
                                      paddle.to_tensor(float("inf")))

        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)
        return output_memory, output_anchors

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           memory_mask=None,
                           denoising_class=None,
                           denoising_bbox=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        output_memory, output_anchors = self._get_encoder_output_anchors(
            memory, spatial_shapes, memory_mask)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(
            output_memory) + output_anchors

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1),
            self.num_queries,
            axis=1,
            sorted=self.topk_sorted)

        # _, topk_ind = paddle.topk(
        #     enc_outputs_class.max(-1),
        #     min(self.num_queries, memory.shape[1]),
        #     axis=1)

        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)
        topk_coords_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                             topk_ind)  # unsigmoided.
        reference_points = enc_topk_bboxes = F.sigmoid(topk_coords_unact)
        if denoising_bbox is not None:
            reference_points = paddle.concat([denoising_bbox, enc_topk_bboxes],
                                             1)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind).detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points.detach(
        ), enc_topk_bboxes, enc_topk_logits
