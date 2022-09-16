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

from turtle import forward
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_, uniform_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS

import paddle
import paddle.nn as nn

import math


def get_act(name):
    '''get_act
    '''
    if name == 'silu':
        return nn.Silu()

    else:
        raise RuntimeError('')


class PHead(nn.Layer):
    __inject__ = ['loss']

    def __init__(self,
                 in_channels_list=[256, 512, 1024],
                 strides_list=[8, 16, 32],
                 hidden_dim=256,
                 num_classes=80,
                 num_proposals_list=[100, 100, 100],
                 data_fmt='row_first',
                 use_head_stem=False,
                 loss='DETRLoss',
                 act='silu'):

        super().__init__()
        self.data_fmt = data_fmt
        self.num_proposals_list = num_proposals_list
        self.num_classes = num_classes
        self.strides_list = strides_list
        self.use_head_stem = use_head_stem
        self.loss = loss

        if use_head_stem:
            self.proposal_stems = nn.LayerList([
                nn.Sequential(
                    nn.Conv2D(c, hidden_dim, 3, 1, 1),
                    nn.BatchNorm2D(hidden_dim),
                    get_act(act), ) for c in in_channels_list
            ])

        self.proposal_convs = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(hidden_dim
                          if use_head_stem else c, hidden_dim, 3, 1, 1),
                nn.BatchNorm2D(hidden_dim),
                get_act(act),
                nn.Conv2D(hidden_dim, hidden_dim, 3, 1, 1),
                nn.BatchNorm2D(hidden_dim),
                get_act(act),
                nn.Conv2D(hidden_dim, num_proposals_list[i], 1, 1), )
            for i, c in enumerate(in_channels_list)
        ])

        self.proposal_pools = nn.LayerList([
            nn.AdaptiveMaxPool2D(
                1, return_mask=True) for _ in in_channels_list
        ])

        self.cls_pred = nn.Sequential(
            nn.Linear(hidden_dim, self.num_classes),
            # nn.Sigmoid()
        )

        self.box_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(), nn.Linear(hidden_dim, 4), nn.Sigmoid())

        # self.init_weights()

    def init_weights(self, ):
        bound = 1 / math.sqrt(self.cls_pred.weight.shape[0])
        uniform_(self.cls_pred.weight, -bound, bound)
        uniform_(self.cls_pred.bias, -bound, bound)

    def forward(self, feats, inputs=None):

        flatten_feats = []
        flatten_coords = []

        for i, feat in enumerate(feats):
            if self.use_head_stem:
                feat = self.proposal_stems[i](feat)

            n, _, h, w = feat.shape

            x = self.proposal_convs[i](feat)
            _, indices = self.proposal_pools[i](x)

            # indices = self.format_global_maxpool2d_index(indices, h, w, data_fmt=self.data_fmt)
            # indices = paddle.to_tensor(indices)
            # b_indices = paddle.tile(paddle.arange(n)[:, None], repeat_times=(self.num_proposals_list[i], )).reshape([-1, 1])
            # indices = paddle.concat([b_indices, indices], axis=-1)            

            s = w if self.data_fmt == 'row_first' else h
            indices = paddle.concat(
                [indices // s, indices % s], axis=-2).squeeze(-1)
            b_indices = paddle.tile(
                paddle.arange(
                    n, dtype='int32')[:, None],
                repeat_times=(self.num_proposals_list[i], )).reshape([-1, 1])
            p_indices = paddle.concat(
                [b_indices, indices.reshape([-1, 2])], axis=-1)

            feat = paddle.gather_nd(feat.transpose([0, 2, 3, 1]),
                                    p_indices).reshape(
                                        [n, self.num_proposals_list[i], -1])

            flatten_feats.append(feat)
            flatten_coords.append(indices * self.strides_list[i])

        flatten_feats = paddle.concat(flatten_feats, axis=1)
        flatten_coords = paddle.concat(flatten_coords, axis=1)

        pred_cls = self.cls_pred(flatten_feats)
        pred_box = self.box_pred(flatten_feats)

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            return self.loss(pred_box[None], pred_cls[None], inputs['gt_bbox'],
                             inputs['gt_class'])

        else:
            return (pred_box, pred_cls, None)

    @staticmethod
    def format_global_maxpool2d_index(index, h, w, data_fmt='row_first'):
        '''
        '''
        assert data_fmt in ('row_first', 'col_first'), ''

        n, c, _, _ = index.shape

        s = w if data_fmt == 'row_first' else h

        output = []

        def func(v):
            jj, ii = divmod(v.item(), s)
            return [jj, ii]

        for i in range(n):
            for j in range(c):
                v = index[i, j, 0, 0]

                output.append(func(v))

        return output


# m = PHead(in_channels_list=[2, ], hidden_dim=3, num_proposals=3)
# feats = [paddle.rand([2, 2, 4, 4]) for i in range(1)]
# m(feats)
