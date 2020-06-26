from flood_forecast.trainer import train_function
from datetime import datetime
import unittest
import os


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "test_config")

    def test_trainer(self):
        pass


if __name__ == '__main__':
    unittest.main()
