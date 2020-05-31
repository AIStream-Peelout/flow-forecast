from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict


def calculate_confidence_intervals(df: pd.DataFrame, ci_lower: float, ci_upper) -> pd.DataFrame:
    assert 0.0 <= ci_lower <= 0.5
    assert 0.5 <= ci_upper <= 1.0
    assert ci_lower != ci_upper
    return df.quantile(q=[ci_lower, ci_upper], axis=1).T


def plot_df_test_with_confidence_interval(
        df_test: pd.DataFrame,
        df_prediction_samples: pd.DataFrame,
        forecast_start_index: int,
        params: Dict,
        ci: float = 95.0,
        alpha=0.25) -> plt.Axes:
    assert 0.0 <= ci <= 100.0
    assert 0.0 < alpha < 1.0
    fig, ax = plt.subplots()
    df_test[["preds", params["dataset_params"]["target_col"][0]]].plot.line(ax=ax)
    ax.axvline(x=forecast_start_index)
    ci_lower, ci_upper = ((100.0 - ci) / 2.0) / 100.0, ((100.0 - ci) / 2.0 + ci) / 100.0
    df_quantiles = calculate_confidence_intervals(df_prediction_samples, ci_lower, ci_upper)
    print("plotting with CI now")
    ax.fill_between(
        df_quantiles.index,
        df_quantiles[ci_lower],
        df_quantiles[ci_upper],
        alpha=alpha)
    return ax
