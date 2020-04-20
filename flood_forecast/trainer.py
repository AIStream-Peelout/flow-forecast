import argparse
from typing import Sequence, List, Tuple, Dict
import json
import wandb
from flood_forecast.pytorch_training import train_transformer_style
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.pre_dict import interpolate_dict
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import evaluate_model
from flood_forecast.pre_dict import scaler_dict

def train_function(model_type: str, params:Dict):
    """
    Function to train a Model(TimeSeriesModel) or da_rnn. Will return the trained model
    model_type str: Type of the model (for now) must be da_rnn or 
    :params dict: Dictionary containing all the parameters needed to run the model
    """
    dataset_params = params["dataset_params"]
    if model_type == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        preprocessed_data = make_data(params["dataset_params"]["training_path"], params["dataset_params"]["target_col"], params["dataset_params"]["forecast_length"])
        config, model = da_rnn(preprocessed_data, len(dataset_params["target_col"]))
        # All train functions return trained_model
        trained_model = train(model, preprocessed_data, config)
    elif model_type == "PyTorch":
        trained_model = PyTorchForecast(params["model_name"], dataset_params["training_path"], dataset_params["validation_path"], dataset_params["test_path"], params)
        train_transformer_style(trained_model, params["training_params"], params["forward_params"])
        params["inference_params"]["dataset_params"]["scaling"] = scaler_dict[dataset_params["scaler"]]
        test_acc = evaluate_model(trained_model, model_type, params["dataset_params"]["target_col"], params["metrics"], params["inference_params"], {})
        wandb.run.summary["test_accuracy"] = test_acc[0]
        test_plot = test_acc[1][["preds", params["dataset_params"]["target_col"][0]]].plot.line()
        # Log plots
        wandb.log({"test_plot":test_plot})
        wandb.log({"test_plot_all": test_acc[1][params["dataset_params"]["relevant_cols"]].plot.line()})
    else: 
        print("Please supply valid model type")
    return trained_model 

def main():
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-p", "--params", help="Path to model config file")
    args = parser.parse_args()
    with open(args.params) as f: 
        training_config = json.load(f)
    train_function(training_config["model_type"], training_config)
    # evaluate_model(trained_model)
    print("Process is now complete.")
if __name__ == "__main__":
    main()

# Example command python flood_forecast/trainer.py -t sample_config.json
