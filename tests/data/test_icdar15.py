import unittest
from detectron2.data.datasets.icdar15 import load_icdar15_instances

class TestICDAR15(unittest.TestCase):

    def setUp(self):
        print('setUp...')

    def test(self):
        img_list_dicts = load_icdar15_instances('/home/appuser/detectron2_data/datasets/ICDAR2015', 'train')

        self.assertEqual(len(img_list_dicts), 1000)

        img_dict = img_list_dicts[0]
        self.assertIn('file_name', img_dict.keys())
        self.assertIn('height', img_dict.keys())
        self.assertIn('width', img_dict.keys())
        self.assertIn('image_id', img_dict.keys())
        self.assertIn('annotations', img_dict.keys())

        ann = img_dict['annotations'][0]
        self.assertIn('bbox', ann.keys())
        self.assertIn('bbox_mode', ann.keys())
        self.assertIn('category_id', ann.keys())

        self.assertEqual(len(ann['bbox']), 5)

    def tearDown(self):
        print('tearDown...')
