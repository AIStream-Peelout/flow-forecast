# flake8: noqa
import argparse
from typing import Dict
import json
import plotly.graph_objects as go
import wandb
import pandas as pd
from flood_forecast.pytorch_training import train_transformer_style
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import evaluate_model
from flood_forecast.time_model import scaling_function
from flood_forecast.plot_functions import (
    plot_df_test_with_confidence_interval,
    plot_df_test_with_probabilistic_confidence_interval)

def handle_model_evaluation1(trained_model, params: Dict, model_type: str) -> None:
    """Utility function to help handle model evaluation. Primarily used at the moment for forcast

    :param trained_model: A PyTorchForecast model that has already been trained. 
    :type trained_model: PyTorchForecast
    :param params: A dictionary of the trained model parameters.
    :type params: Dict
    :param model_type: The type of model. Almost always PyTorch in practice.
    :type model_type: str
    """
    test_acc = evaluate_model(
            trained_model,
            model_type,
            params["dataset_params"]["target_col"],
            params["metrics"],
            params["inference_params"],
            {})
    wandb.run.summary["test_accuracy"] = test_acc[0]
    df_train_and_test = test_acc[1]
    forecast_start_idx = test_acc[2]
    df_prediction_samples = test_acc[3]
    mae = (df_train_and_test.loc[forecast_start_idx:, "preds"] -
            df_train_and_test.loc[forecast_start_idx:, params["dataset_params"]["target_col"][0]]).abs()
    inverse_mae = 1 / mae
    i = 0
    for df in df_prediction_samples:
        pred_std = df.std(axis=1)
        average_prediction_sharpe = (inverse_mae / pred_std).mean()
        wandb.log({'average_prediction_sharpe' + str(i): average_prediction_sharpe})
        i += 1
    df_train_and_test.to_csv("temp_preds.csv")
    # Log plots now
    if "probabilistic" in params["inference_params"]:
        test_plot = plot_df_test_with_probabilistic_confidence_interval(
            df_train_and_test,
            forecast_start_idx,
            params,)
    elif len(df_prediction_samples) > 0:
        for thing in zip(df_prediction_samples, params["dataset_params"]["target_col"]):
            thing[0].to_csv(thing[1] + ".csv")
            test_plot = plot_df_test_with_confidence_interval(
                df_train_and_test,
                thing[0],
                forecast_start_idx,
                params,
                targ_col=thing[1],
                ci=95,
                alpha=0.25)
            wandb.log({"test_plot_" + thing[1]: test_plot})
    else:
        pd.options.plotting.backend = "plotly"
        t = params["dataset_params"]["target_col"][0]
        test_plot = df_train_and_test[[t, "preds"]].plot()
        wandb.log({"test_plot_" + t: test_plot})
    print("Now plotting final plots")
    test_plot_all = go.Figure()
    for relevant_col in params["dataset_params"]["relevant_cols"]:
        test_plot_all.add_trace(
            go.Scatter(
                x=df_train_and_test.index,
                y=df_train_and_test[relevant_col],
                name=relevant_col))
    wandb.log({"test_plot_all": test_plot_all})

def train_function(model_type: str, params: Dict) -> PyTorchForecast:
    """Function to train a Model(TimeSeriesModel) or da_rnn. Will return the trained model
    
    :param model_type: Type of the model. In almost all cases this will be 'PyTorch'
    :type model_type: str
    :param params: Dictionary containing all the parameters needed to run the model
    :type Dict:
    :return: A trained model
    
    .. code-block:: python 
        
        with open("model_config.json") as f: 
            params_dict = json.load(f)
        train_function("PyTorch", params_dict)

    ...

    For information on what this params_dict should include see `Confluence pages <https://flow-forecast.atlassian.net/wiki/spaces/FF/pages/92864513/Getting+Started>`_ on training models. 
    """
    dataset_params = params["dataset_params"]
    if model_type == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        preprocessed_data = make_data(
            params["dataset_params"]["training_path"],
            params["dataset_params"]["target_col"],
            params["dataset_params"]["forecast_length"])
        config, model = da_rnn(preprocessed_data, len(dataset_params["target_col"]))
        # All train functions return trained_model
        trained_model = train(model, preprocessed_data, config)
    elif model_type == "PyTorch":
        dataset_params["batch_size"] = params["training_params"]["batch_size"]
        trained_model = PyTorchForecast(
            params["model_name"],
            dataset_params["training_path"],
            dataset_params["validation_path"],
            dataset_params["test_path"],
            params)
        class2 = False if trained_model.params["dataset_params"]["class"] != "GeneralClassificationLoader" else True
        takes_target = False
        if "takes_target" in trained_model.params:
            takes_target = trained_model.params["takes_target"]

        if "inference_params" in trained_model.params:
            if "dataset_params" not in trained_model.params["inference_params"]:
                print("Using generic dataset params")
                trained_model.params["inference_params"]["dataset_params"] = trained_model.params["dataset_params"].copy()
                del trained_model.params["inference_params"]["dataset_params"]["class"]
                # noqa: F501
                trained_model.params["inference_params"]["dataset_params"]["interpolate_param"] = trained_model.params["inference_params"]["dataset_params"].pop("interpolate")
                trained_model.params["inference_params"]["dataset_params"]["scaling"] = trained_model.params["inference_params"]["dataset_params"].pop("scaler")
                if "feature_param" in trained_model.params["dataset_params"]:
                    trained_model.params["inference_params"]["dataset_params"]["feature_params"] = trained_model.params["inference_params"]["dataset_params"].pop("feature_param")
                delete_params = ["num_workers", "pin_memory", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end",
                                "training_path", "validation_path", "test_path", "batch_size"]
                for param in delete_params:
                    if param in trained_model.params["inference_params"]["dataset_params"]:
                        del trained_model.params["inference_params"]["dataset_params"][param]
        train_transformer_style(model=trained_model,
                                training_params=params["training_params"],
                                takes_target=takes_target,
                                forward_params={}, class2=class2)
        if "scaler" in dataset_params and "inference_params" in params:
            if "scaler_params" in dataset_params:
                params["inference_params"]["dataset_params"]["scaling"] = scaling_function({},
                                                                                           dataset_params)["scaling"]
            else:
                params["inference_params"]["dataset_params"]["scaling"] = scaling_function({},
                                                                                           dataset_params)["scaling"]
            params["inference_params"]["dataset_params"].pop('scaler_params', None)
        # TODO Move to other func
        if params["dataset_params"]["class"] != "GeneralClassificationLoader":
            handle_model_evaluation1(trained_model, params, model_type)

    else:
        raise Exception("Please supply valid model type for forecasting or classification")
    return trained_model


def main():
    """
    Main fundection which is called from the command line. Entrypoint for training all TS ML models.
    """
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-p", "--params", help="Path to model config file")
    args = parser.parse_args()
    with open(args.params) as f:
        training_config = json.load(f)
    train_function(training_config["model_type"], training_config)
    print("Process is now complete.")

if __name__ == "__main__":
    main()
