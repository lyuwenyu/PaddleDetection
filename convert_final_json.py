import json
import os

data = json.loads(open('output/bbox.json', 'r').read())

_data = []
for blob in data:
    if blob['score'] > 0.0:
        _data.append(blob)

data_image_ids = set([x['image_id'] for x in data])
_data_image_ids = set([x['image_id'] for x in _data])
no_preds_image_ids = data_image_ids - _data_image_ids

print('no_preds_image_ids, ', len(no_preds_image_ids))

for i in no_preds_image_ids:
    preds = [x for x in data if x['image_id'] == i]
    scors = [x['score'] for x in data if x['image_id'] == i]
    j = scors.index(max(scors))
    _data.append(preds[j])

_data_image_ids = set([x['image_id'] for x in _data])
no_preds_image_ids = data_image_ids - _data_image_ids
print('no_preds_image_ids after, ', len(no_preds_image_ids))

# --------
import glob
imgs = sorted(glob.glob('/paddle/dataset/nfdw/val/*.jpg'))

from collections import Counter
imgs_id = [x['image_id'] for x in _data]
xx = Counter(imgs_id)
for k, v in xx.items():
    if v > 1:
        print(k, v, os.path.basename(imgs[k]).split('.')[0])

with open('/paddle/dataset/nfdw/val_imgID.txt') as f:
    #  {"id": 20230000001, "file_name": "HuXQEyIAeRq70Z6F4gDTOwh9zPnkBmaoCiNb2f8l",}
    lines = f.readlines()
    im_id_map = {}
    for lin in lines:
        lin = eval(lin)
        im_id_map[lin['file_name']] = lin['id']

import os
for x in _data:
    k = os.path.basename(imgs[x['image_id']]).split('.')[0]
    x['image_id'] = im_id_map[k]

_data[-1]

with open('bbox_final_v16.json', 'w') as f:
    output_json = json.dumps(_data)
    f.write(output_json)
"""
1. python tools/eval.py -c configs/rtdetr/nfdw_swin_l.yml -o weights=./output/nfdw_swin_l/best_model.pdparams --save_prediction_only

2. CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c configs/rtdetr/nfdw_swin_l.yml -o weights=./output/swinv1/32.pdparams --infer_dir=/paddle/dataset/nfdw/val/ --draw_threshold=0.0 --save_results=True 

3. CUDA_VISIBLE_DEVICES=5 python tools/infer.py -c configs/rtdetr/nfdw_swin_l.yml -o weights=./output/swinv1/32.pdparams --infer_dir=/paddle/dataset/nfdw/val/ --slice_infer --slice_size 1920 1920 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.01 --match_metric=ios --draw_threshold=0. --save_results=True 

"""

data = json.loads(open('bbox.json', 'r').read())

_data = []
for blob in data:
    if blob['score'] > 0.4:
        _data.append(blob)

data_image_ids = set([x['image_id'] for x in data])
_data_image_ids = set([x['image_id'] for x in _data])
no_preds_image_ids = data_image_ids - _data_image_ids
print('no_preds_image_ids, ', len(no_preds_image_ids))

for i in no_preds_image_ids:
    preds = [x for x in data if x['image_id'] == i]
    scors = [x['score'] for x in data if x['image_id'] == i]
    j = scors.index(max(scors))
    _data.append(preds[j])

with open('bbox_final_v4.json', 'w') as f:
    output_json = json.dumps(_data)
    f.write(output_json)
