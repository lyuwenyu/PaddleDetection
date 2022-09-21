# https://github.com/lyuwenyu/pytorch_workspace/blob/master/centernet/core/datasets/detdataset.py

import numpy as np


def gaussian_radius(sizes, min_overlap=0.5):
    '''
        sizes: [n, 2], w, h
    '''
    width, height = sizes[:, 0], sizes[:, 1]

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    r = np.concatenate((r1[None], r2[None], r3[None]), axis=0)

    return np.maximum(np.amin(r, axis=0), 0)


def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:
                               radius + right]

    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


class BoxCenterGaussianMask:
    def __init__(
            self,
            min_overlap=0.9, ):
        super().__init__()
        self.min_overlap = min_overlap

    def __call__(self, boxes, shape):

        gt_bbox_list = boxes  # [cx, cy, w, h]

        h, w = shape
        heatmap = np.zeros((len(gt_bbox_list), h, w)).astype(np.float32)
        # print(heatmap.shape)

        for j in range(len(gt_bbox_list)):
            gt_bbox = gt_bbox_list[j]
            gt_bbox = np.array(gt_bbox) * np.array([w, h, w, h])

            if len(gt_bbox) > 0:
                # gt area
                r = gaussian_radius(gt_bbox[:, -2:],
                                    self.min_overlap).astype(np.int32)

                cx = gt_bbox[:, 0].astype(np.int32)
                cy = gt_bbox[:, 1].astype(np.int32)

                # left, right = np.minimum(cx, r), np.minimum(w - cx, r + 1)
                # top, bottom = np.minimum(cy, r), np.minimum(h - cy, r + 1)

                for i in range(len(gt_bbox)):
                    draw_umich_gaussian(heatmap[j, :, :], (cx[i], cy[i]), r[i])

        return heatmap
