import unittest
from flood_forecast.transformer_xl.informer import Informer


class TestInformer(unittest.TestCase):
    def setUp(self):
        self.informer = Informer(5, 5, 2, 10, 10, 1)
