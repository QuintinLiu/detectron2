import unittest
import numpy as np

from detectron2.data.transforms.transform import RotationTransform

class TestRbboxRotationTransform(unittest.TestCase):
    def test_rbbox_rotation(self):
        h, w = 424, 640
        rot = RotationTransform(h, w, 30, expand=True, center=None)

        rbbox = np.array([242.49, 158.95999999999998, 193.42, 173.42, 0])
        rot_rbbox = rot.apply_rotated_box(rbbox[np.newaxis, :])[0]

        print(f'rotated rbbox: {rot_rbbox}')
        self.assertEqual(rot_rbbox.shape[-1], 5)
