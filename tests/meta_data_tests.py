import unittest
import os


class MetaDataTests(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

if __name__ == '__main__':
    unittest.main()
