import unittest
import pandas as pd
import plotly.graph_objects as go
from flood_forecast.plot_functions import calculate_confidence_intervals, plot_df_test_with_confidence_interval


class PlotFunctionsTest(unittest.TestCase):
    """ Tests the plot functions
    """
    df_test = pd.DataFrame({
        'preds': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'target_col': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    })
    df_preds = pd.DataFrame({
        0: [-1.0, -2.0, -1.0, 0.0, -1.0, 6.0],
        1: [1.0, 2.0, 4.0, 3.0, 2.0, 9.0]
    })
    df_preds_empty = pd.DataFrame(index=[0, 1, 2, 3, 4, 5])

    def test_calculate_confidence_intervals(self):
        ci_lower, ci_upper = 0.025, 0.975
        df_quantiles = calculate_confidence_intervals(
            self.df_preds, self.df_test['preds'], ci_lower, ci_upper)
        df_preds_mean = self.df_preds.mean(axis=1)
        self.assertTrue((df_quantiles[ci_lower] < df_preds_mean).all())
        self.assertTrue((df_quantiles[ci_upper] > df_preds_mean).all())
        self.assertTrue((df_quantiles[ci_lower] <= self.df_test['preds']).all())
        self.assertTrue((df_quantiles[ci_upper] >= self.df_test['preds']).all())

    def test_calculate_confidence_intervals_df_preds_empty(self):
        ci_lower, ci_upper = 0.025, 0.975
        df_quantiles = calculate_confidence_intervals(
            self.df_preds_empty, self.df_test['preds'], ci_lower, ci_upper)
        self.assertTrue(df_quantiles[ci_lower].isna().all())
        self.assertTrue(df_quantiles[ci_upper].isna().all())

    def test_plot_df_test_with_confidence_interval(self):
        params = {'dataset_params': {'target_col': ['target_col']}}
        fig = plot_df_test_with_confidence_interval(self.df_test, self.df_preds, 0, params, "target_col", 95)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_df_test_with_confidence_interval_df_preds_empty(self):
        params = {'dataset_params': {'target_col': ['target_col']}}
        fig = plot_df_test_with_confidence_interval(
            self.df_test, self.df_preds_empty, 0, params, "target_col", 95)
        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()
