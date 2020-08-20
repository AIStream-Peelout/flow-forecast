import argparse
from typing import Dict
import json
import plotly.graph_objects as go
import wandb
from flood_forecast.pytorch_training import train_transformer_style
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.evaluator import evaluate_model
from flood_forecast.pre_dict import scaler_dict
from flood_forecast.plot_functions import plot_df_test_w

def train_model(params):
    trained_model = PyTorchForecast(
                params["model_name"],
                dataset_params["training_path"],
                dataset_params["validation_path"],
                dataset_params["test_path"],
                params)
    train_transformer_style(trained_model, params["training_params"], params["forward_params"])

def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and meta models")
    parser.add_argument("-p", )


def if __name__ == "__main__":
    main()