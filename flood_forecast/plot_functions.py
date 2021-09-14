import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch
from typing import Dict, List


def jitter(points: torch.tensor) -> np.ndarray:
    stdev = float(0.01 * (max(points) - min(points)))
    return np.random.randn(len(points)) * stdev


def plot_shap_value_heatmaps(shap_values: torch.tensor,) -> List[go.Figure]:
    average_shap_value_over_batches = shap_values.mean(axis="batches")

    x = [i for i in range(shap_values.align_to("observations", ...).shape[0])]
    y = [i for i in range(shap_values.align_to("preds", ...).shape[0])]

    figs: List[go.Figure] = []
    for shap_values_features in average_shap_value_over_batches.align_to(
        "features", ...
    ):
        fig = go.Figure()
        heatmap = go.Heatmap(
            z=shap_values_features,
            x=x,
            y=y,
            colorbar=dict(title=dict(side="right", text="feature values")),
            colorscale=px.colors.sequential.Bluered,
        )
        fig.add_trace(heatmap)
        fig.update_xaxes(title_text="sequence history steps")
        fig.update_yaxes(title_text="prediction steps")
        figs.append(fig)
    return figs


def plot_summary_shap_values(
    shap_values: torch.tensor, columns: List[str]
) -> go.Figure:
    mean_shap_values = shap_values.mean(axis=["preds", "batches"])

    fig = go.Figure()
    bar_plot = go.Bar(
        y=columns, x=mean_shap_values.abs().mean(axis="observations"), orientation="h"
    )
    fig.add_trace(bar_plot)
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": columns[::-1]})

    return fig


def plot_summary_shap_values_over_time_series(
    shap_values: torch.tensor, columns: List[str]
) -> go.Figure:
    abs_mean_shap_values = shap_values.mean(axis=["batches"]).abs()
    multi_shap_values = abs_mean_shap_values.mean(axis="observations")

    fig = go.Figure()
    for i, pred_shap_values in enumerate(multi_shap_values.align_to("preds", ...)):
        fig.add_trace(
            go.Bar(
                y=columns, x=pred_shap_values, name=f"time-step {i}", orientation="h"
            )
        )
    fig.update_layout(
        barmode="stack",
        yaxis={"categoryorder": "array", "categoryarray": columns[::-1]},
    )
    return fig


def plot_shap_values_from_history(
    shap_values: torch.tensor, history: torch.tensor
) -> List[go.Figure]:
    mean_shap_values = shap_values.mean(axis=["preds", "batches"])
    mean_history_values = history.mean(axis="batches")

    figs: List[go.Figure] = []
    for feature_history, feature_shap_values in zip(
        mean_history_values.align_to("features", ...),
        mean_shap_values.align_to("features", ...),
    ):
        fig = go.Figure()
        scatter = go.Scatter(
            y=jitter(feature_shap_values),
            x=feature_shap_values,
            mode="markers",
            marker=dict(
                color=feature_history,
                colorbar=dict(title=dict(side="right", text="feature values")),
                colorscale=px.colors.sequential.Bluered,
            ),
        )
        fig.add_trace(scatter)
        fig.update_yaxes(range=[-0.05, 0.05])
        fig.update_xaxes(title_text="shap value")
        fig.update_layout(showlegend=False)
        figs.append(fig)
    return figs


def calculate_confidence_intervals(
    df: pd.DataFrame, df_preds: pd.Series, ci_lower: float, ci_upper
) -> pd.DataFrame:
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
    targ_col,
    ci: float = 95.0,
    alpha=0.25,
) -> go.Figure:
    assert 0.0 <= ci <= 100.0
    assert 0.0 < alpha < 1.0
    fig = go.Figure()
    if "pred_" + targ_col in df_test:
        df_test["preds"] = df_test["pred_" + targ_col]
    target_col = targ_col
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test["preds"], name="preds"))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test[target_col], name=target_col))
    ci_lower, ci_upper = (
        ((100.0 - ci) / 2.0) / 100.0,
        ((100.0 - ci) / 2.0 + ci) / 100.0,
    )
    df_quantiles = calculate_confidence_intervals(
        df_prediction_samples, df_test["preds"], ci_lower, ci_upper
    )

    print("plotting with CI now")
    fig.add_trace(
        go.Scatter(
            x=df_quantiles.index.tolist() + df_quantiles.index.tolist()[::-1],
            y=df_quantiles[ci_lower].tolist() + df_quantiles[ci_upper].tolist()[::-1],
            fill="toself",
            name=f"{int(ci)}% confidence interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast_start_index, forecast_start_index],
            y=[
                min(df_quantiles[ci_lower].min(), df_test[target_col].min()),
                max(df_quantiles[ci_lower].max(), df_test[target_col].max()),
            ],
            name="pred_start",
        )
    )
    return fig


def plot_df_test_with_probabilistic_confidence_interval(
    df_test: pd.DataFrame,
    forecast_start_index: int,
    params: Dict,
    real_data=True
) -> go.Figure:
    fig = go.Figure()
    target_col = params["dataset_params"]["target_col"][0]
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test["preds"], name="preds"))
    if real_data:
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test[target_col], name=target_col))
    print("plotting with CI now")
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test["preds"] + 2 * df_test["std_dev"], name="upper bound"))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test["preds"] - 2 * df_test["std_dev"], name="lower bound"))
    fig.add_trace(
        go.Scatter(
            x=[forecast_start_index, forecast_start_index],
            y=[
                min((df_test["preds"] - df_test["std_dev"]).min(), df_test[target_col].min()),
                max((df_test["preds"] + df_test["std_dev"]).max(), df_test[target_col].max()),
            ],
            name="pred_start"
        )
    )
    return fig
