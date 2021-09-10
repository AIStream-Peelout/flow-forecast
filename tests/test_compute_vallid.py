from flood_forecast.pytorch_training import compute_validation
from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader
import unittest


class TestComputeValidation(unittest.TestCase):
    def setUp(self):
        param = {}
        self.classification_loader = GeneralClassificationLoader(param, 2)

    def test_compute_validation(self):
        compute_validation()
