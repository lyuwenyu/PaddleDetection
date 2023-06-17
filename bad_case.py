# HWGTxO7U09BdAqroS3i4JPNCwpFb2MkZunXtjLVa
# MADmYNl8PTQjE03Ke5cds9upvCyZUo1brLHRfBVx

# lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa
"""
from collections import Counter
imgs = [x['image_id'] for x in _data]
xx = Counter(imgs)
for k, v in xx.items():
    if v > 1:
        print(k)

"""
""" swinv1
9 2     n
182 2   p
208 2   p
212 2   p
258 2   n
270 2   p
"""
""" swinv2
use_nms: True
nms_iou_threshold: 0.3
nms_score_threshold: 0.2

106 2   p 
182 2   p
208 2   p 
212 2   p 
# 230 2   n 
231 3   p 
"""
