import argparse
import glob
import json
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import PIL.ImageDraw
from tqdm import tqdm
import cv2

import random


def voc_get_image_info(annotation_root, im_id):
    filename = annotation_root.findtext('filename')

    assert filename is not None
    img_name = os.path.basename(filename)

    size = annotation_root.find('size')
    width = float(size.findtext('width'))
    height = float(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': im_id
    }
    return image_info


def voc_get_coco_annotation(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, "label is not in label2id."
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.findtext('xmin'))
    ymin = float(bndbox.findtext('ymin'))
    xmax = float(bndbox.findtext('xmax'))
    ymax = float(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, "Box size error."
    o_width = xmax - xmin
    o_height = ymax - ymin
    anno = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
    }
    return anno


def voc_xmls_to_cocojson(annotation_paths, label2id, output_file):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # bounding box start id
    im_id = 0
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = voc_get_image_info(ann_root, im_id)
        img_info['file_name'] = os.path.basename(a_path).replace('.xml', '.jpg')

        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = voc_get_coco_annotation(obj=obj, label2id=label2id)
            ann.update({'image_id': im_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        im_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    print(output_json_dict)
    #     output_file = os.path.join(output_dir, output_file)

    with open(output_file, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


if __name__ == '__main__':

    label2id = {'nest': 1, 'kite': 2, 'balloon': 3, 'trash': 4}

    paths = list(glob.glob('./train/*.xml'))
    random.shuffle(paths)

    N = len(paths)
    N_train = int(2 / 3 * N)
    N_val = N - N_train

    voc_xmls_to_cocojson(paths[:N_train], label2id, 'train.json')
    voc_xmls_to_cocojson(paths[-N_val:], label2id, 'val.json')
    voc_xmls_to_cocojson(paths, label2id, 'total.json')