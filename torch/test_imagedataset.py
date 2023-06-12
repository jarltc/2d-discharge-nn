"""
Tests for ImageDataset class
"""

import unittest
from data_helpers import ImageDataset
from pathlib import Path

root = Path.cwd()

class ImageDatasetSquareTest(unittest.TestCase):
    def setUp(self):
        self.dataset = ImageDataset(root/'data'/'interpolation_datasets')

    def tearDown(self):
        del self.dataset
        return super().tearDown()

    def test_something(self):
        result = self.operation()
        self.assertEqual(result, 2)


if __name__ == '__main__':
    unittest.main()