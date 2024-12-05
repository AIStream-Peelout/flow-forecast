import os
import unittest

from sklearn.preprocessing import StandardScaler

from flood_forecast.preprocessing.multimodal_loaders import UniformMultiModalLoader


class TestUniformMultiModalLoader(unittest.TestCase):
    def setUp(self):
        core_parameters = {
            "file_path": "test_data2/multimodal_data.csv",
            "forecast_history": 10,
            "forecast_length": 5,
            "target_col": "cfs",
            "relevant_cols": ["cfs", "height"],
            "scaling": StandardScaler(),
            "interpolate_param": False

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
