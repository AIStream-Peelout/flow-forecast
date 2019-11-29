import argparse
from typing import Sequence, List, Tuple, Dict
import json

def train_function(model:str, training_file_dir:str, test_hours:int, target_col:List[str], wandb:bool = False,
                model_type:str = "PyTorch", **kwargs):
    """
    Function to train a Model(TimeSeriesModel) or da_rnn will return the trained model
     """
    training_param_dict = {}
    training_param_dict["weight_path"] = None
    training_param_dict["learning_rate"] = .09
    training_param_dict["tensorboard_path"] = None
    training_param_dict["max_epochs"] = 2
    training_param_dict["early_stopping"] = False
    for key, value in kwargs.items():
        training_param_dict[key] = value 
    if model == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        preprocessed_data = make_data(training_file_dir, target_col, test_hours)
        config, model = da_rnn(preprocessed_data, len(target_col))
        # All train functions return trained_model
        trained_model = train(model, preprocessed_data, config)
    elif model_type == "PyTorch":
        pass 
    return trained_model 

def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-t", "--train_config", help="Path to model config file")
    args = parser.parse_args()
    with open(args.c) as f: 
      training_config = json.loads(f)
    trained_model = train_function(training_config["model"])
if __name__ == "__main__":
    main()

# Example command python flood_forecast/trainer.py --dataset data/flow_data/concord_final.csv --model da_rnn --task flow --columns cfs
