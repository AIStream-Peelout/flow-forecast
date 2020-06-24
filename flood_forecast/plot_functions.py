import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Iterator
from flood_forecast.named_dimension_array import NamedDimensionArray


def plot_shap_value_heatmaps(shap_values_to_plot: Iterator[NamedDimensionArray], columns: List[str]) -> go.Figure:
    fig = make_subplots(rows=len(columns), subplot_titles=columns)
    cbar_locations = np.linspace(0.15, 0.85, len(columns))
    cbar_len = 1.0 / len(columns)
    for i, (feature_name, shap_values_features) in enumerate(zip(columns, shap_values_to_plot)):
        heatmap = go.Heatmap(
            z=shap_values_features,
            colorbar={'len': cbar_len, 'y': cbar_locations[i]}
        )
        fig.add_trace(heatmap, row=i + 1, col=1)
        fig.update_xaxes(title_text='sequence history steps', row=i + 1, col=1)
        fig.update_yaxes(title_text='prediction steps', row=i + 1, col=1)
    return fig


def calculate_confidence_intervals(df: pd.DataFrame, df_preds: pd.Series, ci_lower: float, ci_upper) -> pd.DataFrame:
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
