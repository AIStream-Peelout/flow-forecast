import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch
from typing import Dict, List


def jitter(points: torch.tensor) -> np.ndarray:
    """
    Adds small, random noise (jitter) to an array of points for visualization purposes.
    The magnitude of the jitter is proportional to the range of the input points.

    :param points: A PyTorch tensor of data points.
     :type points: torch.tensor
     :return: An array of random jitter values with the same length as `points`.
      :rtype: np.ndarray
    """
    stdev = float(0.01 * (max(points) - min(points)))
    return np.random.randn(len(points)) * stdev


def plot_shap_value_heatmaps(shap_values: torch.tensor,) -> List[go.Figure]:
    """
    Generates a list of heatmaps for SHAP values, visualizing the average SHAP
    value across batches for each feature, observation step, and prediction step.

    :param shap_values: A PyTorch tensor of SHAP values, expected to have dimensions
                        like (observations, batches, preds, features, history_steps).
     :type shap_values: torch.tensor
     :return: A list of Plotly Figure objects, one heatmap for each feature.
      :rtype: List[go.Figure]
    """
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
    """
    Creates a horizontal bar chart summarizing the mean absolute SHAP values
    across predictions, batches, and observations to show overall feature importance.

    :param shap_values: A PyTorch tensor of SHAP values.
     :type shap_values: torch.tensor
     :param columns: A list of strings representing the names of the features.
     :type  columns: List[str]
      :return: A Plotly Figure object displaying the summary bar chart.
      :rtype: go.Figure
    """
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
    """
    Creates a stacked horizontal bar chart showing the mean absolute SHAP values
    per feature, averaged over observations and batches, for each prediction time step.

    :param shap_values: A PyTorch tensor of SHAP values.
     :type shap_values: torch.tensor
     :param columns: A list of strings representing the names of the features.
     :type  columns: List[str]
      :return: A Plotly Figure object displaying the stacked bar chart.
      :rtype: go.Figure
    """
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
    """
    Generates a list of scatter plots comparing SHAP values against the
    corresponding feature values from the input history, one plot per feature.

    :param shap_values: A PyTorch tensor of SHAP values.
     :type shap_values: torch.tensor
     :param history: A PyTorch tensor of feature history values.
     :type  history: torch.tensor
      :return: A list of Plotly Figure objects, one scatter plot for each feature.
      :rtype: List[go.Figure]
    """
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
    """
    Calculates confidence interval bounds from prediction samples, ensuring
    the bounds do not cross the point predictions (clipping).

    :param df: A DataFrame where each row is an observation and columns are prediction samples.
     :type df: pd.DataFrame
     :param df_preds: A Series containing the mean/point predictions for each observation.
     :type  df_preds: pd.Series
     :param ci_lower: The lower quantile to calculate (e.g., 0.025 for 95% CI).
     :type  ci_lower: float
     :param ci_upper: The upper quantile to calculate (e.g., 0.975 for 95% CI).
     :type  ci_upper: float
      :return: A DataFrame with two columns, representing the lower and upper CI bounds.
      :rtype: pd.DataFrame
    """
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
    """
    Plots the true values, point predictions, and a confidence interval based
    on prediction samples for a test set.

    :param df_test: DataFrame containing true values (target_col) and point predictions (preds).
     :type df_test: pd.DataFrame
     :param df_prediction_samples: DataFrame where each column is a prediction sample for the test set.
     :type  df_prediction_samples: pd.DataFrame
     :param forecast_start_index: The index where the forecast period begins, for plotting a vertical line.
     :type  forecast_start_index: int
     :param params: A dictionary of parameters (currently unused in the function body).
     :type  params: Dict
     :param targ_col: The name of the target column.
     :type  targ_col: str
     :param ci: The confidence level as a percentage (e.g., 95.0).
     :type  ci: float
     :param alpha: Transparency level for the confidence interval fill (currently unused in the function body).
     :type  alpha: float
      :return: A Plotly Figure object displaying the time series plot with CI.
      :rtype: go.Figure
    """
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
    """
    Plots the true values, point predictions, and a probabilistic confidence interval
    (e.g., mean $\pm$ 2 * standard deviation) for a test set.

    :param df_test: DataFrame containing point predictions ('preds') and standard deviations ('std_dev').
     :type df_test: pd.DataFrame
     :param forecast_start_index: The index where the forecast period begins, for plotting a vertical line.
     :type  forecast_start_index: int
     :param params: A dictionary containing 'dataset_params' with the 'target_col' name.
     :type  params: Dict
     :param real_data: Boolean indicating whether to plot the true values from the target column.
     :type  real_data: bool
      :return: A Plotly Figure object displaying the time series plot with probabilistic CI.
      :rtype: go.Figure
    """
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