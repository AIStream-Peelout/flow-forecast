import plotly.graph_objects as go
import pandas as pd
from typing import Dict


def calculate_confidence_intervals(
        df: pd.DataFrame,
        df_preds: pd.Series,
        ci_lower: float,
        ci_upper) -> pd.DataFrame:
    assert 0.0 <= ci_lower <= 0.5
    assert 0.5 <= ci_upper <= 1.0
    assert ci_lower != ci_upper
    df_quantiles = df.quantile(q=[ci_lower, ci_upper], axis=1).T
    df_quantiles.loc[df_quantiles[ci_lower] > df_preds, ci_lower] = df_preds
    df_quantiles.loc[df_quantiles[ci_upper] < df_preds, ci_upper] = df_preds
    return df_quantiles


def plot_df_test_with_confidence_interval(
        df_test: pd.DataFrame,
        df_prediction_samples: pd.DataFrame,
        forecast_start_index: int,
        params: Dict,
        ci: float = 95.0,
        alpha=0.25) -> go.Figure:
    assert 0.0 <= ci <= 100.0
    assert 0.0 < alpha < 1.0
    fig = go.Figure()

    target_col = params["dataset_params"]["target_col"][0]
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['preds'], name='preds'))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test[target_col], name=target_col))
    ci_lower, ci_upper = ((100.0 - ci) / 2.0) / 100.0, ((100.0 - ci) / 2.0 + ci) / 100.0
    df_quantiles = calculate_confidence_intervals(
        df_prediction_samples,
        df_test['preds'],
        ci_lower,
        ci_upper)

    print("plotting with CI now")
    fig.add_trace(go.Scatter(
        x=df_quantiles.index.tolist() + df_quantiles.index.tolist()[::-1],
        y=df_quantiles[ci_lower].tolist() + df_quantiles[ci_upper].tolist()[::-1],
        fill='toself',
        name=f'{int(ci)}% confidence interval'))
    fig.add_trace(go.Scatter(
        x=[forecast_start_index, forecast_start_index],
        y=[
            min(df_quantiles[ci_lower].min(), df_test[target_col].min()),
            max(df_quantiles[ci_lower].max(), df_test[target_col].max())
        ],
        name='pred_start'))
    return fig
