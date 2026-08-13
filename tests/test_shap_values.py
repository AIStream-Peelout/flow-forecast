import unittest

import numpy as np
import torch

from flood_forecast.explain_model_output import fix_shap_values


class ShapValueShapeTests(unittest.TestCase):
    def setUp(self):
        self.primary_history = torch.zeros(2, 5, 3)
        self.temporal_history = torch.zeros(2, 5, 4)

    def test_current_multi_input_multi_output_format(self):
        shap_values = [
            np.zeros((2, 5, 3, 6)),
            np.ones((2, 5, 4, 6)),
        ]

        normalized = fix_shap_values(
            shap_values, [self.primary_history, self.temporal_history]
        )

        self.assertEqual(normalized.shape, (6, 2, 5, 3))
        self.assertTrue(np.all(normalized == 0))

    def test_current_multi_input_single_output_format(self):
        shap_values = [
            np.zeros((2, 5, 3)),
            np.ones((2, 5, 4)),
        ]

        normalized = fix_shap_values(
            shap_values, [self.primary_history, self.temporal_history]
        )

        self.assertEqual(normalized.shape, (1, 2, 5, 3))

    def test_legacy_multi_input_multi_output_format(self):
        shap_values = [
            [np.zeros((2, 5, 3)), np.ones((2, 5, 4))],
            [np.full((2, 5, 3), 2), np.full((2, 5, 4), 3)],
        ]

        normalized = fix_shap_values(
            shap_values, [self.primary_history, self.temporal_history]
        )

        self.assertEqual(normalized.shape, (2, 2, 5, 3))
        self.assertTrue(np.all(normalized[1] == 2))

    def test_current_single_input_multi_output_format(self):
        shap_values = np.zeros((2, 5, 3, 6))

        normalized = fix_shap_values(shap_values, self.primary_history)

        self.assertEqual(normalized.shape, (6, 2, 5, 3))


if __name__ == "__main__":
    unittest.main()
