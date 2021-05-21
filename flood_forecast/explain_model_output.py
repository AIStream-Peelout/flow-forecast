import random
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import shap
import torch

import wandb
from flood_forecast.plot_functions import (
    plot_shap_value_heatmaps,
    plot_shap_values_from_history,
    plot_summary_shap_values,
    plot_summary_shap_values_over_time_series,
)
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader

BACKGROUND_BATCH_SIZE = 5


def handle_dl_output(dl, dl_class: str, datetime_start: datetime, device: str) -> Tuple[torch.Tensor, int]:
    """

    :param dl: The test data-loader. Should be passed directly
    :type dl: Union[CSVTestLoader, TemporalTestLoader]
    :param dl_class: A string that is the name of DL passef from the params file.
    :type dl_class: str
    :param datetime_start: The start datetime for the forecast
    :type datetime_start: datetime
    :param device: Typical device should be either cpu or cuda
    :type device: str
    :return: Returns a tuple containing either a..
    :rtype: Tuple[torch.Tensor, int]
    """
    if dl_class == "TemporalLoader":
        device = "cpu"
        his, tar, _, forecast_start_idx = dl.get_from_start_date(datetime_start)
        t = tar[1].unsqueeze(0).to(device)
        t1 = tar[0].unsqueeze(0).to(device)
        history = [his[0].unsqueeze(0).to(device), his[1].unsqueeze(0).to(device), t,
                   t1]
    else:
        history, _, forecast_start_idx = dl.get_from_start_date(datetime_start)
        history = history.to(device).unsqueeze(0)
    return history, forecast_start_idx


def _prepare_background_tensor(
    csv_test_loader: CSVTestLoader, backgound_batch_size: int = BACKGROUND_BATCH_SIZE
) -> torch.Tensor:
    """Generate background batches for deep explainer.
    Random sample batches as background data
    background tensor of size (batch_size, history_len, num_feature)
    Args:
        csv_test_loader (CSVTestLoader): test data loader
        backgound_batch_size (int): number of batches used as background data
        for deep explainer. Default to BACKGROUND_BATCH_SIZE.
    Returns:
        torch.Tensor: background tensor of size
        (batch_size, history_len, num_feature)
    """
    background_data = csv_test_loader.original_df
    background_batches = csv_test_loader.convert_history_batches(
        csv_test_loader.df.columns, background_data
    )
    # remove last batch in the list because it may not be of
    # size (history_len, num_feature) due to length of original df
    background_tensor = torch.stack(
        random.sample(background_batches[:-1], backgound_batch_size)
    ).float()
    return background_tensor


def deep_explain_model_summary_plot(
    model, csv_test_loader: CSVTestLoader, datetime_start: Optional[datetime] = None
) -> None:
    """Generate feature summary plot for trained deep learning models
    Args:
        model (object): trained model
        csv_test_loader (CSVTestLoader): test data loader
        datetime_start (datetime, optional): start date of the test prediction,
            Defaults to None, i.e. using model inference parameters.
    """
    if model.params["model_name"] == "SimpleTransformer":
        print("SimpleTransformer currently not supported.")
        return

    use_wandb = model.wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.params["model_name"] == "DARNN" and device.type == "cuda":
        print("DARNN does not work with shap on CUDA")
        return

    if datetime_start is None:
        datetime_start = model.params["inference_params"]["datetime_start"]

    history, forecast_start_idx = handle_dl_output(csv_test_loader, model.params["dataset_params"]["class"],
                                                   datetime_start, device)
    background_tensor = _prepare_background_tensor(csv_test_loader)
    background_tensor = background_tensor.to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    s_values_list = []
    if isinstance(history, list):
        model.model = model.model.to("cpu")
        deep_explainer = shap.DeepExplainer(model.model, history)
        shap_values = deep_explainer.shap_values(history)
        s_values_list.append(shap_values)
    else:
        deep_explainer = shap.DeepExplainer(model.model, background_tensor)
        shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = fix_shap_values(shap_values, history)
    shap_values = np.stack(shap_values)
    # shap_values needs to be 4-dimensional
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )

    # summary plot shows overall feature ranking
    # by average absolute shap values
    fig = plot_summary_shap_values(shap_values, csv_test_loader.df.columns)
    abs_mean_shap_values = shap_values.mean(axis=["preds", "batches"])
    multi_shap_values = abs_mean_shap_values.mean(axis="observations")
    if use_wandb:
        wandb.log({"Overall feature ranking by shap values": fig})
        for idx, col in enumerate(csv_test_loader.df.columns):
            wandb.log({"shap_value_" + col: multi_shap_values})

    # summary plot for multi-step outputs
    # multi_shap_values = shap_values.apply_along_axis(np.mean, 'batches')
    fig = plot_summary_shap_values_over_time_series(
        shap_values, csv_test_loader.df.columns
    )
    if use_wandb:
        wandb.log({"Overall feature ranking per prediction time-step": fig})

    # summary plot for one prediction at datetime_start
    if isinstance(history, list):
        hist = history[0]
    else:
        hist = history

    history_numpy = torch.tensor(
        hist.cpu().numpy(), names=["batches", "observations", "features"]
    )

    shap_values = deep_explainer.shap_values(history)
    shap_values = fix_shap_values(shap_values, history)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )

    figs = plot_shap_values_from_history(shap_values, history_numpy)
    if use_wandb:
        for fig, feature in zip(figs, csv_test_loader.df.columns.tolist()):
            wandb.log(
                {
                    "Feature ranking for prediction"
                    f" at {datetime_start} - {feature}": fig
                }
            )


def fix_shap_values(shap_values, history):
    if isinstance(history, list):
        shap_values = list(zip(*shap_values))[0]
        return shap_values
    return shap_values


def deep_explain_model_heatmap(
    model, csv_test_loader: CSVTestLoader, datetime_start: Optional[datetime] = None
) -> None:
    """Generate feature heatmap for prediction at a start time
    Args:
        model ([type]): trained model
        csv_test_loader ([CSVTestLoader]): test data loader
        datetime_start (Optional[datetime], optional): start date of the test prediction,
            Defaults to None, i.e. using model inference parameters.
    Returns:
        None
    """
    if model.params["model_name"] == "SimpleTransformer":
        print("SimpleTransformer currently not supported.")
        return
    elif "probabilistic" in model.params:
        print("Probabilistic currently not supported.")
        return
    use_wandb = model.wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.params["model_name"] == "DARNN" and device.type == "cuda":
        # TO-DO check if this is still true
        print("Currently DARNN doesn't work with shap on CUDA")
        return

    if datetime_start is None:
        datetime_start = model.params["inference_params"]["datetime_start"]

    history, forecast_start_idx = handle_dl_output(csv_test_loader, model.params["dataset_params"]["class"],
                                                   datetime_start, device)
    background_tensor = _prepare_background_tensor(csv_test_loader)
    background_tensor = background_tensor.to(device)
    model.model.eval()

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    # for each element in each N x M batch in L,
    # attribute to each prediction in forecast len
    s_values_list = []
    if isinstance(history, list):
        deep_explainer = shap.DeepExplainer(model.model, history)
        shap_values = deep_explainer.shap_values(history)
        s_values_list.append(shap_values)
    else:
        deep_explainer = shap.DeepExplainer(model.model, background_tensor)
        shap_values = deep_explainer.shap_values(background_tensor)
    shap_values = fix_shap_values(shap_values, history)
    shap_values = np.stack(shap_values)  # forecast_len x N x L x M
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )
    figs = plot_shap_value_heatmaps(shap_values)
    if use_wandb:
        for fig, feature in zip(figs, csv_test_loader.df.columns):
            wandb.log({f"Average prediction heatmaps - {feature}": fig})

    # heatmap one prediction sequence at datetime_start
    # (seq_len*forecast_len) per fop feature
    to_explain = history
    shap_values = deep_explainer.shap_values(to_explain)
    shap_values = fix_shap_values(shap_values, history)
    shap_values = np.stack(shap_values)
    if len(shap_values.shape) != 4:
        shap_values = np.expand_dims(shap_values, axis=0)
    shap_values = torch.tensor(
        shap_values, names=["preds", "batches", "observations", "features"]
    )  # no fake ballo t
    figs = plot_shap_value_heatmaps(shap_values)
    if use_wandb:
        for fig, feature in zip(figs, csv_test_loader.df.columns):
            wandb.log(
                {"Heatmap for prediction " f"at {datetime_start} - {feature}": fig}
            )
