import random
from datetime import datetime
from typing import Optional

import numpy as np
import shap
import torch

import wandb
from flood_forecast.named_dimension_array import NamedDimensionArray
from flood_forecast.plot_functions import (
    plot_shap_value_heatmaps, plot_shap_values_from_history,
    plot_summary_shap_values, plot_summary_shap_values_over_time_series)
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader

BACKGROUND_SIZE = 5


def deep_explain_model_summary_plot(
    model, csv_test_loader: CSVTestLoader, datetime_start: Optional[datetime] = None,
) -> None:
    """Generate feature summary plot for trained deep learning models

    Args:
        model (object): trained model
        csv_test_loader (CSVTestLoader): test data
        forecast_start_idx (int): test prediction start index
        history (torch.Tensor): history length tensor of dimension
            (seq_len, num_features)
        datetime_start (datetime, optional): start date of the test prediction,
        this should match forecast_start_idx to refer to the same time stamp
        Defaults to datetime(2018, 9, 22, 0).
    """
    use_wandb = model.wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if datetime_start is None:
        datetime_start = model.params["inference_params"]["datetime_start"]

    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    # select most recent 5 batches prior to forcast starte time as background
    background_start_idx = (
        forecast_start_idx - model.params["model_params"]["seq_len"] * BACKGROUND_SIZE
    )
    background_data = csv_test_loader.original_df.iloc[
        background_start_idx:forecast_start_idx
    ]
    # return batch size = BACKGROUND_SIZE+1
    # remove last empty tensor
    background_batches = csv_test_loader.convert_real_batches(
        csv_test_loader.df.columns, background_data
    )[:-1]
    background_tensor = torch.stack(background_batches).float().to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    deep_explainer = shap.DeepExplainer(model.model, background_tensor)
    shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    # summary plot shows overall feature ranking
    # by average absolute shap values
    fig = plot_summary_shap_values(shap_values, csv_test_loader.df.columns)
    if use_wandb:
        wandb.log({"Overall feature ranking by shap values": fig})

    # summary plot for multi-step outputs
    # multi_shap_values = shap_values.apply_along_axis(np.mean, 'batches')
    fig = plot_summary_shap_values_over_time_series(
        shap_values, csv_test_loader.df.columns
    )
    if use_wandb:
        wandb.log({"Overall feature ranking per prediction time-step": fig})

    # summary plot for one prediction at datetime_start

    history = history.to(device).unsqueeze(0)
    history_numpy = NamedDimensionArray(
        history.cpu().numpy(), ["batches", "observations", "features"]
    )

    shap_values = deep_explainer.shap_values(history)
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    fig = plot_shap_values_from_history(
        shap_values, history_numpy, csv_test_loader.df.columns
    )
    if use_wandb:
        wandb.log(
            {
                "Feature ranking for prediction"
                f" at {datetime_start.strftime('%Y-%m-%d')}": fig
            }
        )


def deep_explain_model_heatmap(
    model, csv_test_loader, datetime_start: Optional[datetime] = None
) -> None:
    use_wandb = model.wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if datetime_start is None:
        datetime_start = model.params["inference_params"]["datetime_start"]

    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    # select most recent 5 batches prior to forcast starte time as background
    background_start_idx = (
        forecast_start_idx - model.params["model_params"]["seq_len"] * BACKGROUND_SIZE
    )
    background_data = csv_test_loader.original_df.iloc[
        background_start_idx:forecast_start_idx
    ]
    # return batch size = BACKGROUND_SIZE+1
    # remove last empty tensor
    background_batches = csv_test_loader.convert_real_batches(
        csv_test_loader.df.columns, background_data
    )[:-1]
    background_tensor = torch.stack(background_batches).float().to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    # for each element in each N x M batch in L,
    # attribute to each prediction in forecast len
    deep_explainer = shap.DeepExplainer(model.model, background_tensor)
    shap_values = deep_explainer.shap_values(
        background_tensor
    )  # forecast_len x N x L x M
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    fig = plot_shap_value_heatmaps(shap_values, csv_test_loader.df.columns)
    if use_wandb:
        wandb.log({"Average prediction heatmaps": fig})

    # heatmap one prediction sequence at datetime_start
    # (seq_len*forecast_len) per fop feature
    to_explain = history.to(device).unsqueeze(0)
    shap_values = deep_explainer.shap_values(to_explain)
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    fig = plot_shap_value_heatmaps(shap_values, csv_test_loader.df.columns)
    if use_wandb:
        wandb.log(
            {"Heatmap for prediction " f"at {datetime_start.strftime('%Y-%m-%d')}": fig}
        )
