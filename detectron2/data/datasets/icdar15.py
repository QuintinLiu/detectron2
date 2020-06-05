import os
import re
import cv2
import numpy as np
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ['load_icdar15_instances', 'register_icdar15']

CLASS_NAMES = ['text', 'diffcult']

def load_icdar15_instances(dirname, split):
    assert 'train' in split or 'test' in split, 'split error, only train or test'
    if 'train' in split:
        image_dir = os.path.join(dirname, 'ch4_training_images')
        gt_dir = os.path.join(dirname, 'ch4_training_localization_transcription_gt')
    else:
        image_dir = os.path.join(dirname, 'ch4_test_images')
        gt_dir = os.path.join(dirname, 'Challenge4_Test_Task1_GT')

    img_dicts = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img_id = re.split('_|\.', img_name)[1]

        img = cv2.imread(img_path, 1)
        height = img.shape[0]
        width = img.shape[1]

        anns = []
        with PathManager.open(os.path.join(gt_dir, 'gt_img_{}.txt'.format(img_id)), encoding='UTF-8-sig') as f:
            try:
                res = np.loadtxt(f, dtype=np.str, delimiter=',', comments='####', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
            except ValueError as e:
                print('ValueError: {}. Come from gt_img_{}.txt'.format(e, img_id))

        if res.ndim == 1:
            res = res[np.newaxis, :]
        for i in range(len(res)):
            assert len(res[i]) == 9, 'cols number error, gt_img_{}.txt, {} != 9'.format(img_id, len(res[i]))
            row = res[i][:8].astype(np.int)
            text = res[i][8]

            # x1, y1, x2, y2, x3, y3, x4, y4
            #  0,  1,  2,  3,  4,  5,  6,  7
            edge1 = np.sqrt((row[0] - row[2])**2 + (row[1] - row[3])**2)
            edge2 = np.sqrt((row[2] - row[4])**2 + (row[3] - row[5])**2)
            angle = 0

            if edge1 > edge2:
                wt = edge1
                ht = edge2
                if row[0] - row[2] != 0:
                    angle = -np.arctan(float(row[1] - row[3]) / float(row[0] - row[2])) / np.pi * 180
                else:
                    angle = 90.0
            else:
                wt = edge2
                ht = edge1
                if row[2] - row[4] != 0:
                    angle = -np.arctan(float(row[3] - row[5]) / float(row[2] - row[4])) / np.pi * 180
                else:
                    angle = 90.0
            cx = float(row[0] + row[4]) / 2
            cy = float(row[1] + row[5]) / 2
            
            anns.append({'bbox':[cx, cy, wt, ht, angle], 'bbox_mode':BoxMode.XYWHA_ABS, 'category_id':0 if '###' not in text else 1})

        img_dict = {'file_name': img_path,
                    'height': height,
                    'width': width,
                    'image_id': img_id,
                    'annotations':anns,
                   }
        img_dicts.append(img_dict)
    
    return img_dicts


def register_icdar15(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_icdar15_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES, dirname=dirname, split=split)
