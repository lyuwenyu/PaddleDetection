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

        if query_masks is None:
            return {'query_selection_loss': paddle.zeros([1], dtype='float32')}

        shapes = [_feat.shape[-2:] for _feat in query_masks]

        gt_masks = []
        n_pos = 0

        for i in range(len(inputs['gt_bbox'])):
            _cent = inputs['gt_bbox'][i][:, :2]
            n_pos += len(_cent)

            _gt_masks_per = []
            for (h, w) in shapes:
                _mask = paddle.zeros([h, w], dtype='float32')
                # TODO
                _mask[paddle.cast(_cent[:, 0] * w, 'int'), paddle.cast(
                    _cent[:, 1] * h, 'int')] = 1
                _mask[paddle.cast(_cent[:, 0] * w, 'int'), paddle.cast(
                    _cent[:, 1] * h + 0.5, 'int')] = 1
                _mask[paddle.cast(_cent[:, 0] * w + 0.5, 'int'), paddle.cast(
                    _cent[:, 1] * h + 0.5, 'int')] = 1
                _mask[paddle.cast(_cent[:, 0] * w + 0.5, 'int'), paddle.cast(
                    _cent[:, 1] * h + 0.5, 'int')] = 1
                _gt_masks_per.append(_mask.flatten())

            gt_masks.append(paddle.concat(_gt_masks_per, axis=0).unsqueeze(0))

        gt_masks = paddle.concat(gt_masks, axis=0)
        query_masks = paddle.concat(
            [x.squeeze(1).flatten(1) for x in query_masks], axis=-1)

        loss = F.binary_cross_entropy_with_logits(
            query_masks, gt_masks, reduction='mean')

        # print(gt_masks.shape, query_masks.shape)
        # print(gt_masks.stop_gradient, query_masks.stop_gradient)
        # print(loss)

        return {'query_selection_loss': loss}


from paddle import Tensor


def box_convert(boxes: Tensor, in_fmt='xyxy', out_fmt='cxcywh'):
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
