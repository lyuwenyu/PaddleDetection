import json
from PIL import Image
import os

raw_data = {'info': '', 'licenses': '', 'categories': [], 'images': []}

#
with open('val_imgID.txt') as f:
    lines = f.readlines()

    for lin in lines:
        _info = eval(lin)
        _info['file_name'] = _info['file_name'] + '.jpg'

        im = Image.open(os.path.join('./val', _info['file_name']))
        w, h = im.size

        _info['width'] = w
        _info['height'] = h

        raw_data['images'].append(_info)

label2id = {'nest': 1, 'kite': 2, 'balloon': 3, 'trash': 4}

for k, v in label2id.items():
    raw_data['categories'].append({'supercategory': '', 'id': v, 'name': k})

with open('test.json', 'w') as f:
    output_json = json.dumps(raw_data)
    f.write(output_json)
