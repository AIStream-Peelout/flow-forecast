from typing import Dict
import shap
import torch
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
from flood_forecast.named_dimension_array import NamedDimensionArray
from flood_forecast.plot_functions import (
    plot_shap_value_heatmaps,
    plot_summary_shap_values,
    plot_summary_shap_values_over_time_series,
    plot_shap_values_from_history,
)

random.seed(0)


BACKGROUND_SAMPLE_SIZE = 5


def deep_explain_model_summary_plot(
    model,
    datetime_start: datetime = datetime(2014, 6, 1, 0),
    test_csv_path: str = None,
    forecast_total: int = 336,
    dataset_params: Dict = {},
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(model.params["dataset_params"]["relevant_cols"])
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        csv_test_loader = model.test_data
    else:
        csv_test_loader = CSVTestLoader(
            test_csv_path,
            forecast_total,
            **dataset_params,
            interpolate=dataset_params["interpolate_param"],
        )
    background_data = csv_test_loader.original_df
    background_batches = csv_test_loader.convert_real_batches(
        csv_test_loader.df.columns, background_data
    )
    background_tensor = (
        torch.stack(random.sample(background_batches, BACKGROUND_SAMPLE_SIZE))
        .float()
        .to(device)
    )
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
    wandb.log({"Overall feature ranking by shap values": fig})

    # summary plot for multi-step outputs
    # multi_shap_values = shap_values.apply_along_axis(np.mean, 'batches')
    fig = plot_summary_shap_values_over_time_series(
        shap_values, csv_test_loader.df.columns
    )
    wandb.log({"Overall feature ranking per prediction time-step": fig})

    # summary plot for one prediction at datetime_start
    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
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
    wandb.log(
        {
            f"Feature ranking for prediction at {datetime_start.strftime('%Y-%m-%d')}": fig
        }
    )
    plt.close()


def deep_explain_model_heatmap(
    model,
    datetime_start: datetime = datetime(2014, 6, 1, 0),
    test_csv_path: str = None,
    forecast_total: int = 336,
    dataset_params: Dict = {},
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(model.params["dataset_params"]["relevant_cols"])
    forecast_len = model.params["model_params"]["output_seq_len"]
    seq_len = model.params["model_params"]["seq_len"]
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        csv_test_loader = model.test_data
    else:
        csv_test_loader = CSVTestLoader(
            test_csv_path,
            forecast_total,
            **dataset_params,
            interpolate=dataset_params["interpolate_param"],
        )
    background_data = csv_test_loader.original_df
    background_batches = csv_test_loader.convert_real_batches(
        csv_test_loader.df.columns, background_data
    )
    background_tensor = (
        torch.stack(random.sample(background_batches, BACKGROUND_SAMPLE_SIZE))
        .float()
        .to(device)
    )
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    # for each element in each N x M batch in L, attribute to each prediction in forecast len
    deep_explainer = shap.DeepExplainer(model.model, background_tensor)
    shap_values = deep_explainer.shap_values(
        background_tensor
    )  # forecast_len x N x L x M
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    fig = plot_shap_value_heatmaps(shap_values, csv_test_loader.df.columns)
    wandb.log({"Average prediction heatmaps": fig})

    # heatmap one prediction sequence at datetime_start
    # (seq_len*forecast_len) per fop feature
    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    to_explain = history.to(device).unsqueeze(0)
    shap_values = deep_explainer.shap_values(to_explain)
    shap_values = np.stack(shap_values)
    shap_values = NamedDimensionArray(
        shap_values, ["preds", "batches", "observations", "features"]
    )

    fig = plot_shap_value_heatmaps(shap_values, csv_test_loader.df.columns)
    wandb.log({f"Heatmap for prediction at {datetime_start.strftime('%Y-%m-%d')}": fig})


def deep_explain_model_sample():
    pass
    # TODO
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_features = len(model.params['dataset_params']['relevant_cols'])

    # if type(datetime_start) == str:
    #     datetime_start = datetime.strptime(datetime_start, "%Y-%m-%d")
    # # If the test dataframe is none use default one supplied in params
    # if test_csv_path is None:
    #     csv_test_loader = model.test_data
    # else:
    #     csv_test_loader = CSVTestLoader(
    #         test_csv_path,
    #         hours_to_forecast,
    #         **dataset_params,
    #         interpolate=dataset_params["interpolate_param"]
    #     )
    # # history, forecast_total_df, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)
    # background_data = csv_test_loader.original_df
    # background_batches = csv_test_loader.convert_real_batches(
    #     csv_test_loader.df.columns, background_data
    #     )
    # # TODO - better stratagy to choose background data
    # background_tensor = torch.stack(background_batches[:-1]).float().to(device)
    # model.model.eval()
    # force plot for a simgle sample (in matplotlib)
    # shap.force_plot(
    #     deep_explainer.expected_value[0],
    #     shap_values[0].reshape(-1, 3)[6, :],
    #     show=True,
    #     feature_names=csv_test_loader.df.columns,
    #     matplotlib=True,
    # )
    # # force plot for multiple time-steps
    # # can only be generated as html objects
    # # shap.force_plot(e.expected_value[0], shap_values[0].reshape(-1, 3), show=True, feature_names=csv_test_loader.df.columns)
    # # dependece plot shows feature value vs shap value
    # shap.dependence_plot(
    #     2,
    #     shap_values[0].reshape(-1, 3),
    #     to_explain.cpu().numpy().reshape(-1, 3),
    #     interaction_index=0,
    #     feature_names=csv_test_loader.df.columns,
    # )
