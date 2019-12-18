import argparse
from typing import Sequence, List, Tuple, Dict
import json
from flood_forecast.pytorch_training import train_transformer_style
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.time_model import PyTorchForecast

def train_function(model_type, params:Dict):
    """
    Function to train a Model(TimeSeriesModel) or da_rnn will return the trained model
     """

    if model_type == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        dataset_params = params["dataset_param"]
        preprocessed_data = make_data(params["dataset_params"]["training_path"], params["dataset_params"]["target_col"], params["dataset_param"]["forecast_length"])
        config, model = da_rnn(preprocessed_data, len(dataset_params["target_col"]))
        # All train functions return trained_model
        trained_model = train(model, preprocessed_data, config)
    elif model_type == "PyTorch":
        model = PyTorchForecast(params["model_params"]["model_base"], dataset_params["training_path"], dataset_params["validation_path"])
        train_transformer_style(model, params, params["wandb"], params["model_params"]["forward_param"])
        
    return trained_model 

def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-t", "--train_config", help="Path to model config file")
    args = parser.parse_args()
    with open(args.c) as f: 
      training_config = json.loads(f)
    trained_model = train_function(training_config["model_type"], training_config)

if __name__ == "__main__":
    main()

# Example command python flood_forecast/trainer.py --dataset data/flow_data/concord_final.csv --model da_rnn --task flow --columns cfs
