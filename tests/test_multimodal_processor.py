import unittest
from flood_forecast.preprocessing.multimodal_loaders import UniformMultiModalLoader


class TestUniformMultiModalLoader(unittest.TestCase):
    def setUp(self):
        core_parameters = {
            "csv_path": "tests/test_data/test_multimodal.csv",
            "forecast_history": 10,
            "forecast_length": 5,


        }
        text_config = {

        }
        image_config = {

        }

        self.multimodal_loader = UniformMultiModalLoader(
            csv_data_loader_params=core_parameters,
            image_config=image_config,
            text_config=text_config
        )

    def test_output_single_row(self):
        pass


if __name__ == '__main__':
    unittest.main()
