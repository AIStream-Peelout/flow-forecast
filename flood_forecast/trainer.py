import argparse 

def train_function(model:str, training_file_dir:str, test_hours:int, target_col:List[str], n_targs=1):
    if model == "da_rnn":
        from flood_forecast.da_rnn.train_da import da_rnn, train
        from flood_forecast.preprocessing.preprocess_da_rnn import make_data
        preprocessed_data = make_data(training_file_dir, target_col, test_hours)
        model = da_rnn(preprocessed_data, len(target_col))
        elif model == "":
            pass 


def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-d", "--dataset", help="Path to river flow csv file")
    parser.add_argument("-m", "--model", help="Model you want to use for training")
    parser.add_argument("-t", "--test", default=336, help="The number of hours to forecast for the test")
    parser.add_argument("-t", "--task", help="The task you want to train the model for")
    parser.add_argument("-c", "--column", default="cfs", help="The target column either height, cfs or both")
    parser.add_argument("")
    if args.column == "both":
        args.column = ['cfs', 'height']
    else: 
        args.column = [args.column]
    
if __name__ == "__main__":
    main()