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
"""
182 2 ibnT3qEg5C40SoJ12ODch9MHuyeQI8LaXRPUzk7d
197 2 jFm2yBOqvw8YJKgUrPMfsDpZlICde367o9kR0nHx n
208 2 jN6p8r4Wtf5UCbLYez3lcD0B1o7KhsVydqT9FJGH
212 2 jWY6SuPVFy7hXQigBLnqCZR8pmk4eD2sOrtHwIEx
231 2 k7hwRcumYlLnQxF9J4fpT0G2MIEONKoBbZrsvdzC n
270 2 lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa
"""
"""
no_preds_image_ids after,  0
21 2 IMHyhtAizuO1v72VJbRcds9XCpxPSk63EYgUZfLo
66 2 JmHgaKjqMQfuA7r5hVv0ZIb469sF2tCw3oDcnklx
182 2 ibnT3qEg5C40SoJ12ODch9MHuyeQI8LaXRPUzk7d
197 2 jFm2yBOqvw8YJKgUrPMfsDpZlICde367o9kR0nHx n
208 2 jN6p8r4Wtf5UCbLYez3lcD0B1o7KhsVydqT9FJGH
212 2 jWY6SuPVFy7hXQigBLnqCZR8pmk4eD2sOrtHwIEx
231 2 k7hwRcumYlLnQxF9J4fpT0G2MIEONKoBbZrsvdzC
250 2 kso3OURFTYeywacpQ7GDzJlvX2MnSrPKxguqdjiL
270 2 lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa


no_preds_image_ids,  0
no_preds_image_ids after,  0
66 2 JmHgaKjqMQfuA7r5hVv0ZIb469sF2tCw3oDcnklx
182 2 ibnT3qEg5C40SoJ12ODch9MHuyeQI8LaXRPUzk7d
208 2 jN6p8r4Wtf5UCbLYez3lcD0B1o7KhsVydqT9FJGH
212 2 jWY6SuPVFy7hXQigBLnqCZR8pmk4eD2sOrtHwIEx
270 2 lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa
"""
"""
no_preds_image_ids after,  0
123 2 LUgwKYyd1cCbXqZ4pD0kFVHv9Bt3T6Oe2ArlPIsN
178 2 iQGwl7gqZmka9T01p5ISAosBytvrb2nzMYWcN8hE
182 2 ibnT3qEg5C40SoJ12ODch9MHuyeQI8LaXRPUzk7d
198 2 jG0mElAJpsbonSquahf1tz7WeZ68Ly29kPHYiIBx
208 2 jN6p8r4Wtf5UCbLYez3lcD0B1o7KhsVydqT9FJGH
212 2 jWY6SuPVFy7hXQigBLnqCZR8pmk4eD2sOrtHwIEx
214 2 jd3Eas5JgcZ0nVelpwkuHT6WF2mDNzBUxC8OGvyA
231 2 k7hwRcumYlLnQxF9J4fpT0G2MIEONKoBbZrsvdzC
260 2 l456jQUt8ebIocXDhy2dEiYsfpRvrPTB0JLVSH7q
270 2 lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa
"""
"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.946
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.999
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.952
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.947
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.946
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.958
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.963
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.959
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.964
[06/16 16:29:04] ppdet.engine INFO: Total sample number: 267, average FPS: 4.9989879438987685
[06/16 16:29:04] ppdet.engine INFO: Best test bbox ap is 0.946.
"""
