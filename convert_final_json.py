import json

data = json.loads(open('bbox_swin.json', 'r').read())

_data = []
for blob in data:
    if blob['score'] > 0.0:
        _data.append(blob)

data_image_ids = set([x['image_id'] for x in data])
_data_image_ids = set([x['image_id'] for x in _data])
no_preds_image_ids = data_image_ids - _data_image_ids

for i in no_preds_image_ids:
    preds = [x for x in data if x['image_id'] == i]
    scors = [x['score'] for x in data if x['image_id'] == i]
    j = scors.index(max(scors))
    _data.append(preds[j])

from collections import Counter
imgs = [x['image_id'] for x in _data]
xx = Counter(imgs)
for k, v in xx.items():
    if v > 1:
        print(k, v)

import glob
imgs = sorted(glob.glob(''))

with open('bbox_final_v1.json', 'w') as f:
    output_json = json.dumps(_data)
    f.write(output_json)
"""
1. python tools/eval.py -c configs/x -o weights=best_model.pdparams --save_prediction_only

2. CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c configs/rtdetr/nfdw_swin_l.yml -o weights=./output/nfdw_swin_l/best_model.pdparams --infer_img=/paddle/dataset/nfdw/val/lJbshc2FqtgIRQHj6dEyLKS34Y7ouNAz1kWCPZpa.jpg --slice_infer --draw_threshold=0.1

"""
