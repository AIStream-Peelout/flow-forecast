from flood_forecast.pytorch_training import compute_validation
import unittest


class TestComputeValidaton(unittest.TestCase):
    def setUp(self):
        self.classification_loader = 1

    def test_compute_validation(self):
        compute_validation()
