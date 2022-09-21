import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr, dtype
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register

import math
import numpy as np
from ppdet.modeling.initializer import bias_init_with_prob, constant_, conv_init_
# from ..backbones.csp_darknet import BaseConv, DWConv
# from ppdet.modeling.losses import IouLoss
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner
from ppdet.modeling.bbox_utils import bbox_overlaps
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PHead']

from ..bbox_utils import bbox_iou

from .heatmap import BoxCenterGaussianMask


class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


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
        self.act = act

        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y


@register
class PHead(nn.Layer):
    __shared__ = ['num_classes', 'width_mult', 'act', 'trt', 'exclude_nms']
    __inject__ = ['assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            width_mult=1.0,
            depthwise=False,
            in_channels=[256, 512, 1024],
            feat_channels=256,
            fpn_strides=(8, 16, 32),
            l1_epoch=300,
            act='silu',
            assigner=SimOTAAssigner(use_vfl=False),
            nms='MultiClassNMS',
            loss_weight={
                'cls': 1.0,
                'obj': 1.0,
                'iou': 5.0,
                'l1': 1.0,
            },
            trt=False,
            exclude_nms=False,
            ppn_threshold=0.1,
            ppn_topk=0.1,
            ppn_select_type='topk',
            ppn_gt_type='center',
            use_obj=True, ):

        super().__init__()
        self._dtype = paddle.framework.get_default_dtype()
        self.num_classes = num_classes
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        feat_channels = int(feat_channels * width_mult)
        self.fpn_strides = fpn_strides
        self.l1_epoch = l1_epoch
        self.assigner = assigner
        self.nms = nms
        self.ppn_threshold = ppn_threshold
        self.ppn_topk = ppn_topk
        self.ppn_select_type = ppn_select_type
        self.use_obj = use_obj
        self.ppn_gt_type = ppn_gt_type

        self.draw_gassian_mask = BoxCenterGaussianMask()

        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.loss_weight = loss_weight
        self.iou_loss = IouLoss(loss_weight=1.0)  # default loss_weight 2.5

        ConvBlock = BaseConv

        self.ppn_convs = nn.LayerList([
            nn.Sequential(
                ConvBlock(
                    c, c, 1, 1, act=act),
                nn.Conv2D(
                    c, 1, 1, bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            for c in self.in_channels
        ])

        self.stem_conv = nn.LayerList()
        self.conv_cls = nn.LayerList()
        self.conv_reg = nn.LayerList()  # reg [x,y,w,h] + obj

        for in_c in self.in_channels:
            self.stem_conv.append(BaseConv(in_c, feat_channels, 1, 1, act=act))

            self.conv_cls.append(
                nn.Sequential(*[
                    ConvBlock(
                        feat_channels, feat_channels, 1, 1, act=act), ConvBlock(
                            feat_channels, feat_channels, 1, 1, act=act),
                    nn.Conv2D(
                        feat_channels,
                        self.num_classes,
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))

            self.conv_reg.append(
                nn.Sequential(*[
                    ConvBlock(
                        feat_channels, feat_channels, 1, 1, act=act),
                    ConvBlock(
                        feat_channels, feat_channels, 1, 1, act=act),
                    nn.Conv2D(
                        feat_channels,
                        4 + 1,  # reg [x,y,w,h] + obj
                        1,
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
                ]))

        # encoder_layer = nn.TransformerEncoderLayer(
        #     hidden_dim, nhead, dim_feedforward, dropout, activation=act)
        # self.encoder = PPTransformerEncoder(
        #     encoder_layer, num_layers, return_intermediate=return_intermediate)

        # https://github.com/lyuwenyu/PaddleDetection/blob/yolo_ctm_L/ppdet/modeling/transformers/tencoder.py

        self._init_weights()

    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_reg = paddle.full([5], math.log(5.), dtype=self._dtype)
        bias_reg[:2] = 0.
        bias_reg[-1] = bias_cls
        for cls_, reg_ in zip(self.conv_cls, self.conv_reg):
            constant_(cls_[-1].weight)
            constant_(cls_[-1].bias, bias_cls)
            constant_(reg_[-1].weight)
            reg_[-1].bias.set_value(bias_reg)

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        # feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]

        cls_score_list, reg_pred_list = [], []
        obj_score_list = []

        pp_logits_list = []
        index_list = []
        stride_list = []
        mask_list = []

        for i, feat in enumerate(feats):
            n, c, h, w = feat.shape
            pp_feat = self.ppn_convs[i](feat)
            pp_logits_list.append(pp_feat)

            if self.training and self.ppn_gt_type == 'gaussian':
                boxes = targets['gt_bbox'][0]
                boxes[:, 2:] -= boxes[:, :2]
                boxes[:, :2] += boxes[:, 2:] / 2.
                mask = self.draw_gassian_mask(boxes, shape=(h, w))
                mask_list.append(paddle.to_tensor(mask))

            if self.ppn_select_type == 'threshod':
                index = (F.sigmoid(pp_feat) > self.ppn_threshold
                         ).squeeze(1).nonzero()

                if len(index) < 10:
                    topk = 10
                    v, index = paddle.topk(
                        F.sigmoid(pp_feat).squeeze(1).flatten(1),
                        sorted=False,
                        k=topk,
                        axis=-1)
                    index = paddle.concat(
                        [
                            paddle.zeros(
                                [topk, ], dtype='int64').unsqueeze(-1),
                            (index[0] // w).unsqueeze(-1),  # h
                            (index[0] % w).unsqueeze(-1),  # w
                        ],
                        axis=-1)

            elif self.ppn_select_type == 'topk':
                # TODO select topk 
                topk = int(
                    h * w *
                    self.ppn_topk) if 0 < self.ppn_topk < 1 else self.ppn_topk
                topk = max(topk, 10)

                v, index = paddle.topk(
                    F.sigmoid(pp_feat).squeeze(1).flatten(1),
                    sorted=False,
                    k=topk,
                    axis=-1)
                index = paddle.concat(
                    [
                        paddle.zeros(
                            [topk, ], dtype='int64').unsqueeze(-1),
                        (index[0] // w).unsqueeze(-1),  # h
                        (index[0] % w).unsqueeze(-1),  # w
                    ],
                    axis=-1)

            index_list.append(index)
            stride_list.append(
                paddle.full(
                    [index.shape[0], 1], self.fpn_strides[i],
                    dtype=self._dtype))

            # just for bs=1
            feat = paddle.gather_nd(
                feat.transpose([0, 2, 3, 1]), index=index).reshape([n, -1, c])

            # TODO add attention

            feat = feat.transpose([0, 2, 1]).unsqueeze(-1)  # N C L1 1

            feat = self.stem_conv[i](feat)
            cls_logit = self.conv_cls[i](feat)
            reg_pred = self.conv_reg[i](feat)

            # cls prediction
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            # reg prediction
            reg_xywh, obj_logit = paddle.split(reg_pred, [4, 1], axis=1)
            reg_xywh = reg_xywh.flatten(2).transpose([0, 2, 1])
            reg_pred_list.append(reg_xywh)
            # obj prediction
            obj_score = F.sigmoid(obj_logit)
            obj_score_list.append(obj_score.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)
        obj_score_list = paddle.concat(obj_score_list, axis=1)

        index_list = paddle.concat(index_list, axis=0)  # L 3
        stride_list = paddle.concat(stride_list, axis=0)

        stride_tensor = stride_list
        anchor_points = paddle.concat(
            [index_list[:, 2:], index_list[:, 1:2]],
            axis=-1) * stride_tensor * 1.0

        anchor_points.stop_gradient = True
        stride_tensor.stop_gradient = True

        # bbox decode
        # anchor_points, stride_tensor, _ =\
        #     self._generate_anchor_point(feat_sizes, self.fpn_strides)

        reg_xy, reg_wh = paddle.split(reg_pred_list, 2, axis=-1)  # N L1 2
        reg_xy += (anchor_points / stride_tensor)

        reg_wh = paddle.exp(reg_wh) * 0.5
        bbox_pred_list = paddle.concat(
            [reg_xy - reg_wh, reg_xy + reg_wh], axis=-1)

        if self.training:
            # anchor_points, stride_tensor, num_anchors_list =\
            #     self._generate_anchor_point(feat_sizes, self.fpn_strides, 0.5)
            num_anchors_list = None
            yolox_losses = self.get_loss([
                cls_score_list, bbox_pred_list, obj_score_list, anchor_points,
                stride_tensor, num_anchors_list
            ], targets)

            # TODO only support bs==1, pp_logtis_list
            loss_pps = 0

            if self.ppn_gt_type == 'center':
                gt_bboxes = paddle.to_tensor(targets['gt_bbox'])[0]  # bs==1
                gt_centers = (
                    gt_bboxes[:, 2:] + gt_bboxes[:, :2]) / 2.  # fix - to +
                for i, pp_logits in enumerate(pp_logits_list):
                    centers = paddle.cast(gt_centers / self.fpn_strides[i],
                                          'int64')
                    pp_gt = paddle.zeros_like(pp_logits)
                    pp_gt[0, 0, centers[:, -1], centers[:, 0]] = 1.
                    loss_pp = F.binary_cross_entropy_with_logits(
                        pp_logits, pp_gt, reduction='mean')
                    loss_pps += loss_pp

            if self.ppn_gt_type == 'gassian':
                for i, pp_logits in enumerate(pp_logits_list):
                    loss_pp = F.binary_cross_entropy_with_logits(
                        pp_logits.squeeze(1), mask_list[i], reduction='mean')
                    loss_pps += loss_pp

            yolox_losses['loss_pps'] = loss_pp
            yolox_losses['loss'] += loss_pps

            return yolox_losses

        else:
            if self.use_obj:
                pred_scores = (cls_score_list * obj_score_list).sqrt()
            else:
                pred_scores = cls_score_list

            return pred_scores, bbox_pred_list, stride_tensor

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, pred_obj,\
        anchor_points, stride_tensor, num_anchors_list = head_outs
        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        if self.use_obj:
            pred_scores = (pred_cls * pred_obj).sqrt()
        else:
            pred_scores = pred_cls

        # label assignment
        center_and_strides = paddle.concat(
            [anchor_points, stride_tensor, stride_tensor], axis=-1)
        pos_num_list, label_list, bbox_target_list = [], [], []
        for pred_score, pred_bbox, gt_box, gt_label in zip(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor, gt_bboxes, gt_labels):
            pos_num, label, _, bbox_target = self.assigner(
                pred_score, center_and_strides, pred_bbox, gt_box, gt_label)
            pos_num_list.append(pos_num)
            label_list.append(label)
            bbox_target_list.append(bbox_target)
        labels = paddle.to_tensor(np.stack(label_list, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_list, axis=0))
        bbox_targets /= stride_tensor  # rescale bbox

        # 1. obj score loss
        mask_positive = (labels != self.num_classes)

        if self.use_obj:
            loss_obj = F.binary_cross_entropy(
                pred_obj,
                mask_positive.astype(pred_obj.dtype).unsqueeze(-1),
                reduction='sum')
        else:
            loss_obj = 0

        num_pos = sum(pos_num_list)

        if num_pos > 0:
            num_pos = paddle.to_tensor(num_pos, dtype=self._dtype).clip(min=1)
            loss_obj /= num_pos

            # 2. iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                bbox_targets, bbox_mask).reshape([-1, 4])
            bbox_iou = bbox_overlaps(pred_bboxes_pos, assigned_bboxes_pos)
            bbox_iou = paddle.diag(bbox_iou)

            loss_iou = self.iou_loss(
                pred_bboxes_pos.split(
                    4, axis=-1),
                assigned_bboxes_pos.split(
                    4, axis=-1))
            loss_iou = loss_iou.sum() / num_pos

            # 3. cls loss
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(
                pred_cls, cls_mask).reshape([-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(labels, mask_positive)
            assigned_cls_pos = F.one_hot(assigned_cls_pos,
                                         self.num_classes + 1)[..., :-1]
            assigned_cls_pos *= bbox_iou.unsqueeze(-1)
            loss_cls = F.binary_cross_entropy(
                pred_cls_pos, assigned_cls_pos, reduction='sum')
            loss_cls /= num_pos

            # 4. l1 loss
            if targets['epoch_id'] >= self.l1_epoch:
                loss_l1 = F.l1_loss(
                    pred_bboxes_pos, assigned_bboxes_pos, reduction='sum')
                loss_l1 /= num_pos
            else:
                loss_l1 = paddle.zeros([1])
                loss_l1.stop_gradient = False
        else:
            loss_cls = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])
            loss_cls.stop_gradient = False
            loss_iou.stop_gradient = False
            loss_l1.stop_gradient = False

        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou

        if targets['epoch_id'] >= self.l1_epoch:
            loss += (self.loss_weight['l1'] * loss_l1)

        yolox_losses = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_obj': loss_obj,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return yolox_losses

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_bboxes, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes *= stride_tensor
        # scale bbox to origin image
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
