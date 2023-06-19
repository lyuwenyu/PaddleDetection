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
from math import floor

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create
import copy
import paddle.nn.functional as F

__all__ = ['DETR']
# Deformable DETR, DINO use the same architecture as DETR


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False,
                 freeze_backbone=False,
                 test_multiscales=None,
                 test_flip=False,
                 num_classes=5,
                 nms_iou_threshold=0.0,
                 final_score_threshold=0.3):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.test_flip = test_flip

        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.final_score_threshold = final_score_threshold
        self.test_multiscales = test_multiscales
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.stop_gradient = True
            self.backbone.eval()
            print('freeze backbone done.')

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
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
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):

        if self.test_multiscales is None:
            return self._forward()

        else:
            return self.get_numtiscale_test_pred()

    @paddle.no_grad()
    def get_numtiscale_test_pred(self):
        # dict_keys(['im_id', 'curr_iter', 'image', 'im_shape', 'scale_factor'])
        self.eval()

        orig_size = self.inputs['image'].shape[-1]
        sf = self.inputs['scale_factor'].flatten().tolist()
        orig_img_h = floor(orig_size / sf[0] + 0.5)
        orig_img_w = floor(orig_size / sf[1] + 0.5)
        flippeds = [False, True] if self.test_flip else [False, ]

        # inputs_list = []
        outputs = []

        for s in self.test_multiscales:

            for is_flipped in flippeds:
                if s == orig_size:
                    # inputs_list.append(self.inputs)
                    # continue
                    inputs = copy.deepcopy(self.inputs)

                else:
                    inputs = {}
                    inputs['image'] = F.interpolate(
                        self.inputs['image'], size=(s, s), mode='bilinear')
                    inputs['im_shape'] = paddle.to_tensor(
                        [[s, s]], dtype='float32')
                    inputs['scale_factor'] = paddle.to_tensor(
                        [[s / (orig_size / sf[0]), s / (orig_size / sf[1])]],
                        dtype='float32')
                    inputs['im_id'] = self.inputs['im_id']

                if is_flipped:
                    inputs['image'] = inputs['image'][:, :, :, ::-1]
                    # inputs['image'] = paddle.flip( inputs['image'], -1)

                # Backbone
                body_feats = self.backbone(inputs)

                # Neck
                if self.neck is not None:
                    body_feats = self.neck(body_feats)

                # Transformer
                pad_mask = inputs.get('pad_mask', None)
                out_transformer = self.transformer(body_feats, pad_mask, inputs)

                preds = self.detr_head(out_transformer, body_feats)
                if self.exclude_post_process:
                    bbox, bbox_num, mask = preds
                else:
                    bbox, bbox_num, mask = self.post_process(
                        preds, inputs['im_shape'], inputs['scale_factor'],
                        paddle.shape(inputs['image'])[2:])

                if is_flipped:
                    # assert orig_img_w >= bbox[:, 2] and  orig_img_w - bbox[:, 4], ''
                    # bbox[:, 2] = orig_img_w - bbox[:, 4] 
                    # bbox[:, 4] = orig_img_w - bbox[:, 2]
                    _bbox = copy.deepcopy(bbox)
                    _bbox[:, 2] = orig_img_w - bbox[:, 4]
                    _bbox[:, 4] = orig_img_w - bbox[:, 2]
                    bbox = _bbox

                output = {'bbox': bbox, 'bbox_num': bbox_num}

                outputs.append(output)

        bbox_preds = []
        for output in outputs:
            bbox_preds.append(output['bbox'].numpy())
        bbox_preds = np.concatenate(bbox_preds)

        bbox_preds = multiclass_nms(
            bbox_preds,
            self.num_classes,
            match_threshold=self.nms_iou_threshold,
            match_metric='iou')

        bbox_preds = np.concatenate(bbox_preds)

        # bbox_preds = paddle.to_tensor(bbox_preds)
        # bbox_num = paddle.to_tensor(len(bbox_preds)) 

        _bbox_preds = bbox_preds[bbox_preds[:, 1] > self.final_score_threshold]
        if len(_bbox_preds) == 0:
            j = np.argmax(bbox_preds[:, 1])
            _bbox_preds = bbox_preds[j]

        bbox_preds = paddle.to_tensor(_bbox_preds)
        bbox_preds = bbox_preds.reshape([-1, 6])

        bbox_num = paddle.to_tensor(len(bbox_preds))

        output = {'bbox': bbox_preds, 'bbox_num': bbox_num}

        return output


import numpy as np


def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='ios'):
    final_boxes = []
    if num_classes == -1:
        idxs = bboxs[:, 0] > num_classes
        keep = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        r = bboxs[idxs, 1:][keep]
        # np.full((r.shape[0], 1), c)
        final_boxes.append(np.concatenate([bboxs[:, :1][keep], r], 1))
    else:
        for c in range(num_classes):
            idxs = bboxs[:, 0] == c
            if np.count_nonzero(idxs) == 0: continue

            keep = nms(bboxs[idxs, 1:], match_threshold, match_metric)
            r = bboxs[idxs, 1:][keep]
            # r = py_softnms(bboxs[idxs, 1:], iou=match_threshold)
            final_boxes.append(
                np.concatenate([np.full((r.shape[0], 1), c), r], 1))

    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    # dets = dets[keep, :]
    return keep


def py_softnms(orig_dets, iou=0.3, sigma=0.5, thresh=0.001, method=0):
    """reference https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
    py_softnms
    :param dets:   boxes, format [x1, y1, x2, y2]
    :param scores:     score
    :param iou:     iou
    :param sigma:  gaussian func
    :param thresh: score thr
    :param method: method
    :return:       keep index
    """
    scores = orig_dets[:, 0]
    dets = orig_dets[:, 1:]

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0

        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > iou] = weight[ovr > iou] - ovr[ovr > iou]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > iou] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep
    # return orig_dets[keep]
