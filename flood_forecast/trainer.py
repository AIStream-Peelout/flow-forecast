import argparse
from typing import Sequence, List, Tuple, Dict

def train_function(model:str, training_file_dir:str, test_hours:int, target_col:List[str], additional_params:Dict=None):
    if model == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        preprocessed_data = make_data(training_file_dir, target_col, test_hours)
        config, model = da_rnn(preprocessed_data, len(target_col))
        train(model, preprocessed_data, config)
    elif model == "":
        pass 


def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-d", "--dataset", help="Path to river flow csv file")
    parser.add_argument("-m", "--model", help="Model you want to use for training")
    parser.add_argument("-t", "--test", default=336, help="The number of hours to forecast for the test")
    parser.add_argument("-b", "--task", help="The task you want to train the model for")
    parser.add_argument("-c", "--column", default="cfs", help="The target column either height, cfs or both")
    parser.add_argument("r", "--resume", default=None, help="Resume from a checkpoint")
    parser.add_argument("-s", "--max_epochs", default="10")
    parser.add_argument("--tensorboard", default="False")
    args = parser.parse_args()
    if args.column == "both":
        args.column = ['cfs', 'height']
    else: 
        args.column = [args.column]
    train_function(args.model, args.dataset, args.test, args.column)
if __name__ == "__main__":
    main()

# Example command python trainer.py --dataset data/flow_data/concord_final.csv --model da_rnn --task flow --columns cfs