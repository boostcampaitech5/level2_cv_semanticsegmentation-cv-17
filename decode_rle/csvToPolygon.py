import os

import numpy as np

import torch
import csv

import numpy as np
from imantics import Polygons, Mask
from random import *

import json


def decode_rle_to_mask(rle, height=2048, width=2048):
    s = rle.split()
    # print('\ns', s)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # print('\nstarts', starts)
    # print('\nlengths', lengths)
    starts -= 1
    ends = starts + lengths
    # print('\nends', ends)
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # print('\nimg', img)
    # print('\nnon_zero', np.count_nonzero(img))
    
    return img.reshape(height, width)


def mask_to_polygon(mask):
    
    # print(mask)
    # print(np.count_nonzero(mask))
    polygons = Mask(mask).polygons()
    
    # print(polygons.points)
    # print(polygons.points[0])
    
    # print(polygons.segmentation)
    return polygons.points[0].tolist()
    


def rle_to_polygon(folder_name, image_size): # folder_name
    print(f'create polygon to ..')
    SAVED_DIR = "/opt/ml/input/code/trained_model/"
    
    f = open(os.path.join(SAVED_DIR, folder_name) + '/output.csv', 'r')
    rdr = csv.reader(f)

    ps_json = {}
    annotations = {
        'annotations' : [],
        "attributes": {},
        "file_id": "-",
        "filename": "-",
        "parent_path": "/X-ray_UNIST/20220824/ID027/USB", 
        "last_modifier_id": "-", 
        "metadata": {"height": 3060, "width": 3060}, 
        "last_workers": {}
    }

    for index, line in enumerate(rdr):
        # print(line)
        if index==0:
            continue
        else:
            image_name = line[0]
            label_name = line[1]
            # print(image_name, label_name)
            if image_name not in ps_json:
                ps_json[image_name] = annotations
                ps_json[image_name]['filename'] = image_name
            mask = decode_rle_to_mask(line[2], image_size[0], image_size[1])
            # print('\n', result, len(result))
            # print(np.count_nonzero(result))
            polygon = mask_to_polygon(mask)
            
            label_data = {
                "id": "ps-id-%d" %randint(1, 1000000000000000),
                "type": "poly_seg",
                "attributes": {},
                "points": polygon,
                "label": label_name
            }
            # print(label_data)
            ps_json[image_name]['annotations'].append(label_data)
            # if len(ps_json) == 3:
            #     print(ps_json.keys())
            #     break
        
    save_ps_json(ps_json)


def save_ps_json(ps_json):
    print(f'save psuedo labeled json..')
    origin_root = "../data/test/DCM"
    ps_root = "../data/test/ps_outputs_json"
    # print(ps_json)

    path = []
    for folder in os.listdir(origin_root):
        for file in os.listdir(origin_root +"/"+ folder):
            if file[-3:].lower() == "png":
                file_path = folder + "/" + file
                path.append(file_path)

    for file_path in path:
        folder_name = file_path.split("/")[0]
        file_name = file_path.split("/")[1]
        os.makedirs(ps_root + "/" + folder_name, exist_ok=True)
        print(folder_name, file_name)

        PS_ROOT = ps_root + "/" + folder_name + "/" + file_name[:-3]+'json'
        print(PS_ROOT)
        with open(PS_ROOT, 'w', encoding='utf-8') as train_writer:
            json.dump(ps_json[file_name], train_writer, indent=2, ensure_ascii=False)
    print("Finish")


if __name__ == "__main__":
    image_size = (2048, 2048)
    folder_name = "[UnetPlusPlus_resnet101]_[size:(512, 512)]_[loss:BCEWithLogitsLoss()]_[LR:0.0001]_[seed:21]_[epoch:200]"
    rle_to_polygon(folder_name, image_size)