import sys
sys.path.append("..")
from flood_forecast.model_dict_function import pytorch_model_dict
import unittest
import os

class TimeSeriesModelTest(unittest.TestCase):
    def setUp(self):
        self.test_path = ""

    def test_pytorch_model_dict(self):
        self.assertEqual(type(pytorch_model_dict), dict)




if __name__ == '__main__':
    unittest.main()