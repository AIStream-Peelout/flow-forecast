import json
import re
import os
import argparse
import traceback
from flood_forecast.trainer import train_function
from typing import List


def split_on_letter(s: str) -> List:
    match = re.compile(r"[^\W\d]").search(s)
    return [s[:match.start()], s[match.start():]]


def loop_through(
        data_dir: str,
        interrmittent_gcs: bool = False,
        use_transfer: bool = True,
        start_index: int = 0,
        end_index: int = 25) -> None:
    """
    Function that makes and executes a set of config files
    This is since we have over 9k files and.
    """
    if not os.path.exists("model_save"):
        os.mkdir("model_save")
    sorted_dir_list = sorted(os.listdir(data_dir))
    # total = len(sorted_dir_list)
    for i in range(start_index, end_index):
        file_name = sorted_dir_list[i]
        station_id_gage = file_name.split("_flow.csv")[0]
        res = split_on_letter(station_id_gage)
        gage_id = res[0]
        station_id = res[1]
        file_path_name = os.path.join(data_dir, file_name)
        print("Training on: " + file_path_name)
        correct_file = None
        if use_transfer and len(os.listdir("model_save")) > 1:
            weight_files = filter(lambda x: x.endswith(".pth"), os.listdir("model_save"))
            paths = []
            for weight_file in weight_files:
                paths.append(os.path.join("model_save", weight_file))
            correct_file = max(paths, key=os.path.getctime)
            print(correct_file)
        config = make_config_file(file_path_name, gage_id, station_id, correct_file)
        extension = ".json"
        file_name_json = station_id + "config_f" + extension
        with open(file_name_json, "w+") as f:
            json.dump(config, f)
        try:
            train_function("PyTorch", config)
        except Exception as e:
            print("An exception occured for: " + file_name_json)
            traceback.print_exc()
            print(e)


def make_config_file(flow_file_path: str, gage_id: str, station_id: str, weight_path: str = None):
    the_config = {
        "model_name": "MultiAttnHeadSimple",
        "model_type": "PyTorch",
        # "weight_path": "31_December_201906_32AM_model.pth",
        "model_params": {
            "number_time_series": 3,
            "seq_len": 36
        },
        "dataset_params":
        {"class": "default",
         "training_path": flow_file_path,
         "validation_path": flow_file_path,
         "test_path": flow_file_path,
         "batch_size": 20,
         "forecast_history": 36,
         "forecast_length": 36,
         "train_end": 35000,
         "valid_start": 35001,
         "valid_end": 40000,
         "target_col": ["cfs1"],
         "relevant_cols": ["cfs1", "precip", "temp"],
         "scaler": "StandardScaler"
         },
        "training_params":
        {
            "criterion": "MSE",
            "optimizer": "Adam",
            "optim_params": {
                "lr": 0.001
                # Default is lr=0.001
            },

            "epochs": 14,
            "batch_size": 20
        },
        "GCS": True,

        "wandb": {
            "name": "flood_forecast_" + str(gage_id),
            "tags": [gage_id, station_id, "MultiAttnHeadSimple", "36", "corrected"]
        },
        "forward_params": {}
    }
    if weight_path:
        the_config["weight_path"] = weight_path
        # 31_December_201906_12AM_model.pth
    return the_config


def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and evaluation")
    parser.add_argument("-p", "--path", help="Data path")

if __name__ == "__main__":
    main()
