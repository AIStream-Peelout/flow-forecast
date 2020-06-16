from typing import Dict
import shap
import torch
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader


def deep_explain_model_summary_plot(
    model,
    datetime_start: datetime = datetime(2018, 9, 22, 0),
    test_csv_path: str = None,
    hours_to_forecast: int = 336,
    dataset_params: Dict = {}
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(datetime_start) == str:
        datetime_start = datetime.strptime(datetime_start, '%Y-%m-%d')
    # If the test dataframe is none use default one supplied in params
    if test_csv_path is None:
        csv_test_loader = model.test_data
    else:
        csv_test_loader = CSVTestLoader(
            test_csv_path,
            hours_to_forecast,
            **dataset_params,
            interpolate=dataset_params["interpolate_param"]
        )

    # TODO - figure out how to get_from_start_date, use history as test set to generate shap
    # the history here doesn't look quite write
    history, _, forecast_start_idx = csv_test_loader.get_from_start_date(datetime_start)

    model.model.eval()


    # csv_test_load_iter = iter(csv_test_loader)
    # csv_list = [i[0] for i in csv_test_load_iter]
    # background = torch.cat(csv_list)

    # background shape (L, N, M)
    # L - batch size, N - history length, M - feature size
    background = history.to(device).unsqueeze(0)
    deep_explainer = shap.DeepExplainer(model.model, background)
    shap_values = deep_explainer.shap_values(background)

    # summary plot shows overall feature ranking by average absolute shap values
    mean_shap_values = np.concatenate(shap_values).mean(axis=0)
    shap.summary_plot(mean_shap_values, tests.cpu().numpy().reshape(-1, 3), feature_names=csv_test_loader.df.columns, plot_type="bar")
    # force plot for a simgle sample (in matplotlib)
    shap.force_plot(e.expected_value[0], shap_values[0].reshape(-1, 3)[6,:], show=True, feature_names=csv_test_loader.df.columns, matplotlib=True)
    # force plot for multiple time-steps 
    # can only be generated as html objects
    # shap.force_plot(e.expected_value[0], shap_values[0].reshape(-1, 3), show=True, feature_names=csv_test_loader.df.columns)
    # dependece plot shows feature value vs shap value
    shap.dependence_plot(2, shap_values[0].reshape(-1, 3), tests.cpu().numpy().reshape(-1, 3), interaction_index=0, feature_names=csv_test_loader.df.columns)
    