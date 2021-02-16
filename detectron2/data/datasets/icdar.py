import os
import re
import cv2
import math
import numpy as np
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ['load_icdar15_instances', 'register_icdar15']


def draw_abcd_rbbox(img, rbbox):
    for i in range(0, 8, 2):
        j = (i + 2) % 8
        img = cv2.line(img, (round(rbbox[i]), round(rbbox[i+1])), 
                        (round(rbbox[j]), round(rbbox[j+1]))
                        , (0, 255, 0), thickness=1)
    return img
def draw_1_cxcywha_rbbox(img, rbbox):
    cnt_x, cnt_y, w, h, angle = rbbox
    theta = angle * math.pi / 180.0 
    c = math.cos(theta)
    s = math.sin(theta)
    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
    rotated_rect = [(int(round(s * yy + c * xx + cnt_x)), int(round(c * yy - s * xx + cnt_y)))\
                    for (xx, yy) in rect]
    for i in range(4):
        j = (i + 1) % 4
        img = cv2.line(img, rotated_rect[i], rotated_rect[j], (0, 0, 255)
                        , thickness=1)
    return img


def process_txt_to_anns(gt_txt):
    """
    process txt content to anns which detectron2 needed

    gt_txt: gt loaded from icdar txt file

    return: per image anns: list[dict], dict include keys "bbox", "BoxMode", "category_id" at least

    """
    anns = []

    if gt_txt.ndim == 1:
        gt_txt = gt_txt[np.newaxis, :]
    gt_txt = gt_txt.astype(np.int)

    if gt_txt.shape[1] == 4:
        gt_txt = gt_txt[:, :, np.newaxis]
        gt_txt = np.hstack((gt_txt[:, 0], gt_txt[:, 1], gt_txt[:, 2], gt_txt[:, 1], gt_txt[:, 2], gt_txt[:, 3], gt_txt[:, 0], gt_txt[:, 3]))

    assert gt_txt.ndim == 2 and gt_txt.shape[1] == 8, f'gt_txt ndim {gt_txt.ndim} and shape {gt_txt.shape} error'

    for i in range(gt_txt.shape[0]):
        row = gt_txt[i]

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
        
        anns.append({'bbox':[cx, cy, wt, ht, angle], 'bbox_mode':BoxMode.XYWHA_ABS, 'category_id':0})

        #img = draw_abcd_rbbox(img, row)
        #img = draw_1_cxcywha_rbbox(img, (cx, cy, wt, ht, angle)) 

    return anns


def load_icdar15_instances(dirname, split):
    assert 'train' == split or 'test' == split, 'split error, only train or test'
    if 'train' == split:
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

        with PathManager.open(os.path.join(gt_dir, 'gt_img_{}.txt'.format(img_id)), encoding='UTF-8-sig') as f:
            try:
                res = np.loadtxt(f, dtype=np.str, delimiter=',', comments='####', usecols=(0, 1, 2, 3, 4, 5, 6, 7))
            except ValueError as e:
                print('ValueError: {}. Come from gt_img_{}.txt'.format(e, img_id))

        anns = process_txt_to_anns(res)
        #cv2.imwrite('/home/appuser/detectron2_data/dt2_exp/results/{}.png'.format(img_id), img)

        img_dict = {'file_name': img_path,
                    'height': height,
                    'width': width,
                    'image_id': img_id,
                    'annotations':anns,
                   }
        img_dicts.append(img_dict)
    
    return img_dicts

def load_icdar13_instances(dirname, split):
    assert 'train' == split or 'test' == split, 'split error, only train or test'
    if 'train' == split:
        image_dir = os.path.join(dirname, 'Challenge2_Training_Task12_Images')
        gt_dir = os.path.join(dirname, 'Challenge2_Training_Task1_GT')
    else:
        image_dir = os.path.join(dirname, 'Challenge2_Test_Task12_Images')
        gt_dir = os.path.join(dirname, 'Challenge2_Test_Task1_GT')

    img_dicts = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img_id = re.split('_|\.', img_name)[1]

        img = cv2.imread(img_path, 1)
        height = img.shape[0]
        width = img.shape[1]

        with PathManager.open(os.path.join(gt_dir, 'gt_img_{}.txt'.format(img_id)), encoding='UTF-8-sig') as f:
            try:
                res = np.loadtxt(f, dtype=np.str, delimiter=',', comments='####', usecols=(0, 1, 2, 3))
            except ValueError as e:
                print('ValueError: {}. Come from gt_img_{}.txt'.format(e, img_id))

        anns = process_txt_to_anns(res)
        #cv2.imwrite('/home/appuser/detectron2_data/dt2_exp/results/{}.png'.format(img_id), img)

        img_dict = {'file_name': img_path,
                    'height': height,
                    'width': width,
                    'image_id': img_id,
                    'annotations':anns,
                   }
        img_dicts.append(img_dict)
    
    return img_dicts


def register_icdar15(name, dirname, split):
    CLASS_NAMES = ['text']
    DatasetCatalog.register(name, lambda: load_icdar15_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES, dirname=dirname, split=split)


def register_icdar13(name, dirname, split):
    CLASS_NAMES = ['text']
    DatasetCatalog.register(name, lambda: load_icdar13_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES, dirname=dirname, split=split)
