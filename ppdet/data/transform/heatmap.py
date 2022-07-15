import math
import numpy as np

from .operators import BaseOperator
from .operators import register_op


@register_op
class BoxCenterGaussianMask(BaseOperator):
    def __init__(self, min_overlap=0.9):
        super().__init__()
        self.min_overlap = min_overlap

    def apply(self, sample, context=None):
        gt_bbox = sample['gt_bbox']

        image = sample['image']
        h, w, _ = image.shape

        heatmap = np.zeros((h, w), dtype=np.float32)

        if len(gt_bbox) > 0:

            r = gaussian_radius(gt_bbox[:, -2:],
                                self.min_overlap).astype(np.int32)

            cx = gt_bbox[:, 0].astype(np.int32)
            cy = gt_bbox[:, 1].astype(np.int32)

            left, right = np.minimum(cx, r), np.minimum(w - cx, r + 1)
            top, bottom = np.minimum(cy, r), np.minimum(h - cy, r + 1)

            for i in range(len(gt_bbox)):
                heatmap[cy[i] - top[i]:cy[i] + bottom[i], cx[i] - left[i]:cx[i]
                        + right[i]] = 1

        sample['heatmap'] = heatmap

        return sample


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
