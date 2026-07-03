"""Local smoke test: exercise the NARX notebook config path (train + multi-step inference)."""
from flood_forecast.trainer import train_function
from flood_forecast.evaluator import infer_on_torch_model

FH, FL = 48, 24
path = "tests/test_data/keag_small.csv"
config = {
    "model_name": "NARX",
    "model_type": "PyTorch",
    "model_params": {
        "n_time_series": 3, "forecast_history": FH, "output_seq_len": FL,
        "n_targets": 1, "n_target_lags": FH, "n_exog_lags": FH,
        "hidden_size": 64, "num_hidden_layers": 2, "dropout": 0.1, "activation": "tanh",
    },
    "dataset_params": {
        "class": "default", "training_path": path, "validation_path": path, "test_path": path,
        "batch_size": 64, "forecast_history": FH, "forecast_length": FL,
        "train_start": 1, "train_end": 5000, "valid_start": 5001, "valid_end": 6000,
        "test_start": 6000, "test_end": 7000,
        "target_col": ["cfs"], "relevant_cols": ["cfs", "precip", "temp"],
        "scaler": "StandardScaler", "interpolate": False,
    },
    "training_params": {
        "criterion": "MSE", "optimizer": "Adam", "optim_params": {},
        "lr": 0.001, "epochs": 1, "batch_size": 64,
    },
    "GCS": False, "wandb": False, "forward_params": {}, "metrics": ["MSE"],
    "inference_params": {
        "datetime_start": "2016-05-31", "hours_to_forecast": 100, "test_csv_path": path,
        "decoder_params": {"decoder_function": "simple_decode", "unsqueeze_dim": 1},
        "dataset_params": {
            "file_path": path, "forecast_history": FH, "forecast_length": FL,
            "relevant_cols": ["cfs", "precip", "temp"], "target_col": ["cfs"],
            "scaling": "StandardScaler", "interpolate_param": False,
        },
    },
}

m = train_function("PyTorch", config)
print("TRAINED OK")
df, end_tensor, hist, idx, loader, samples = infer_on_torch_model(m, **config["inference_params"])
print("INFER OK; df cols:", [c for c in df.columns if c in ("cfs", "preds")], "rows:", len(df))
print("preds non-zero count:", int((df["preds"] != 0).sum()))
