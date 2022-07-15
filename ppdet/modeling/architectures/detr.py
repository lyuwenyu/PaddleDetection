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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import nonzero

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sympy import re

from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DETR']


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 detr_head,
                 post_process='DETRBBoxPostProcess'):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # transformer
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        *out_transformer, query_masks = self.transformer(
            body_feats, self.inputs['pad_mask'])

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)

            query_selection_losses = self.get_location_loss(query_masks)

            return { ** detr_losses, ** query_selection_losses}

        else:
            preds = self.detr_head(out_transformer, body_feats)
            bbox, bbox_num = self.post_process(preds, self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            return bbox, bbox_num

    def get_loss(self, ):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output

    def get_location_loss(self, query_masks):
        '''
        inputs: ['im_id', 'is_crowd', 'gt_class', 'gt_bbox', 'curr_iter', 'image', 'im_shape', 'scale_factor', 'pad_mask', 'epoch_id']
        inputs['gt_bbox']: list, [13, 4], cxcywh, normalized  [[0.07688655, 0.49768999, 0.07883899, 0.27166003], ...]

        query_masks, [[1, 1, 80, 80], [1, 1, 40, 40], [1, 1, 20, 20]]
        im_shape, 
        - NormalizeBox: {}
        - BboxXYXY2XYWH: {}
        '''
        inputs = self.inputs

        # print(inputs.keys())
        # print(inputs['gt_bbox'][0].shape)
        # print(inputs['gt_bbox'][0])
        # print([x.shape for x in query_masks])
        # print(inputs['im_shape'])
        # print(inputs['heatmap'].shape)

        if query_masks is None:
            return {'query_selection_loss': paddle.zeros([1], dtype='float32')}

        shapes = [_feat.shape[-2:] for _feat in query_masks]

        gt_masks = []
        n_pos = 0

        if 'heatmap' in inputs:
            heatmap = inputs['heatmap'][:, None]
            # print('heatmap', heatmap.shape)
            heatmaps = [
                F.interpolate(
                    heatmap, size=m.shape[-2:], mode='nearest')
                for m in query_masks
            ]
            # print('heatmaps', [m.shape for m in heatmaps])
            # print('query_masks', [m.shape for m in query_masks])
            # print('heatmaps', [m.sum() for m in heatmaps])
            gt_masks = paddle.concat(
                [x.squeeze(1).flatten(1) for x in heatmaps], axis=-1)

        else:

            for i in range(len(inputs['gt_bbox'])):

                _cents = inputs['gt_bbox'][i][:, :2]
                n_pos += len(_cents)

                _gt_masks_per = []
                for (h, w) in shapes:
                    _mask = paddle.zeros([h, w], dtype='float32')
                    # TODO

                    if len(_cents) > 0:
                        _points = []
                        for _pt in [(0, 0), (0, 0.5), (0.5, 0), (0.5, 0.5)]:
                            _pts = _cents * paddle.to_tensor(
                                [w, h]) + paddle.to_tensor(_pt)
                            _points.append(paddle.cast(_pts, 'int'))

                        _points = paddle.concat(_points, axis=0)
                        _points[:, 0] = _points[:, 0].clip(0, w - 1)
                        _points[:, 1] = _points[:, 1].clip(0, h - 1)
                        _mask[_points[:, 1], _points[:, 0]] = 1.

                    # _mask[paddle.cast(_cents[:, 1] * h, 'int'), paddle.cast(
                    #     _cents[:, 0] * w, 'int')] = 1

                    # _mask[paddle.cast(_cents[:, 1] * h, 'int'),
                    #       paddle.cast(_cents[:, 0] * w, 'int')] = 1

                    # _mask[paddle.cast(_cents[:, 1] * h + 0.5, 'int').clip(0, h - 1),
                    #       paddle.cast(_cents[:, 0] * w, 'int')] = 1

                    # _mask[paddle.cast(_cents[:, 1] * h, 'int'),
                    #       paddle.cast(_cents[:, 0] * w + 0.5, 'int').clip(0, w -
                    #                                                      1)] = 1

                    # _mask[paddle.cast(_cents[:, 1] * h + 0.5, 'int').clip(0, h - 1),
                    #       paddle.cast(_cents[:, 0] * w + 0.5, 'int').clip(0, w -
                    #                                                      1)] = 1

                    _gt_masks_per.append(_mask.flatten())

                gt_masks.append(
                    paddle.concat(
                        _gt_masks_per, axis=0).unsqueeze(0))

            gt_masks = paddle.concat(gt_masks, axis=0)

        query_masks = paddle.concat(
            [x.squeeze(1).flatten(1) for x in query_masks], axis=-1)

        # loss = F.binary_cross_entropy_with_logits(
        #     query_masks, gt_masks, reduction='mean', ) 

        loss = F.binary_cross_entropy_with_logits(
            query_masks,
            gt_masks,
            reduction='none', ) * ((gt_masks == 0) * 1. + gt_masks * 10.)

        loss = loss.mean() * (gt_masks == 1).sum()

        # loss = binary_focal_loss_with_logits(query_masks, gt_masks)
        # loss = binary_focal_loss_with_logits(query_masks, gt_masks) / (
        #     n_pos + 1)

        # print(gt_masks.shape, query_masks.shape)
        # print(gt_masks.stop_gradient, query_masks.stop_gradient)
        # print(loss)

        return {'query_selection_loss': loss}


import paddle.nn.functional as F


def binary_focal_loss_with_logits(logits, label, alpha=0.25, gamma=2.0):
    score = F.sigmoid(logits)

    weight = (score - label).pow(gamma)
    if alpha > 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        weight *= alpha_t
    loss = F.binary_cross_entropy(score, label, weight=weight, reduction='sum')

    return loss


def box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh'):
    '''boxes convert
    '''
    if in_fmt == out_fmt:
        return boxes

    if in_fmt == 'xyxy' and out_fmt == 'cxcywh':
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        return paddle.stack((cx, cy, w, h), axis=-1)

    elif in_fmt == 'cxcywh' and out_fmt == 'xyxy':
        cx, cy, w, h = boxes.unbind(-1)
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return paddle.stack((x1, y1, x2, y2), axis=-1)

    else:
        raise AttributeError('')


# 6265   File "/root/paddlejob/workspace/env_run/lvwenyu01/PaddleDetection/ppdet/modeling/architectures/detr.py", line 80, in
# 6266     query_selection_losses = self.get_location_loss(query_masks)
# 6267   File "/root/paddlejob/workspace/env_run/lvwenyu01/PaddleDetection/ppdet/modeling/architectures/detr.py", line 144, i
# 6268     _pts = _cent * paddle.to_tensor([w, h]) + paddle.to_tensor(_pt)
# 6269   File "/root/anaconda3/lib/python3.8/site-packages/paddle/fluid/dygraph/math_op_patch.py", line 264, in __impl__
# 6270     return math_op(self, other_var, 'axis', axis)
# 6271 RuntimeError: (PreconditionNotMet) The Tensor's element number must be equal or greater than zero. The Tensor's shape
# 6272   [Hint: Expected numel() >= 0, but received numel():-2 < 0:0.] (at /paddle/paddle/fluid/framework/tensor.cc:59)
# 6273   [operator < elementwise_mul > error]
