"""Helper script that generates NARX_Virgin_Predict.ipynb. Run once, then the .ipynb is the artifact."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Virgin River Forecasting with NARX\n"
    "\n"
    "In this notebook we forecast river flows on the Virgin River in Utah using the **NARX** "
    "(Nonlinear AutoRegressive with eXogenous inputs) model in Flow Forecast. The Virgin River "
    "runs through Zion National Park, where flash floods can be dangerous to hikers, so accurate "
    "short-term forecasts matter.\n"
    "\n"
    "NARX predicts the next target value(s) from a window of past target values (the autoregressive "
    "lags) plus a window of past exogenous drivers (precipitation, temperature, dew point). It is a "
    "lightweight MLP, so it trains quickly and is a strong baseline. At inference time predictions are "
    "fed back into the input window (closed-loop) via `simple_decode`."
))

cells.append(nbf.v4.new_markdown_cell("## 1. Setup: clone Flow Forecast and download data"))

cells.append(nbf.v4.new_code_cell(
    "from google.colab import auth\n"
    "import os\n"
    "auth.authenticate_user()\n"
    "\n"
    "!git clone https://github.com/AIStream-Peelout/flow-forecast.git\n"
    "# Single-gage Virgin River file used for this demo\n"
    "!gsutil cp gs://aistream-datasets/flowdb/09405500AZC_flow.csv ."
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Prepare the data\n"
    "\n"
    "We drop rows missing the timestamp, flow (`cfs`) or precipitation (`p01m`), add a `change_cfs` "
    "first-difference column (sometimes easier to learn than the raw level), and write the cleaned "
    "frame back to disk for the data loader."
))

cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "\n"
    "df = pd.read_csv(\"09405500AZC_flow.csv\", low_memory=False)\n"
    "df = df.dropna(subset=[\"hour_updated\", \"cfs\", \"p01m\"])\n"
    "\n"
    "# The raw timestamps are timezone-aware (e.g. 2014-04-11 16:00:00+00:00). Flow Forecast's\n"
    "# loaders cast the sort/datetime columns with .astype(\"datetime64[ns]\"), which raises\n"
    "# \"cannot supply both a tz and a timezone-naive dtype\" on tz-aware values. We normalize to\n"
    "# UTC and drop the timezone up front so both the train and test loaders can parse them.\n"
    "for col in [\"hour_updated\", \"datetime\"]:\n"
    "    if col in df.columns:\n"
    "        df[col] = pd.to_datetime(df[col], utc=True, errors=\"coerce\").dt.tz_localize(None)\n"
    "df = df.dropna(subset=[\"hour_updated\"])\n"
    "\n"
    "df = df.sort_values(by=[\"hour_updated\"]).reset_index(drop=True)\n"
    "df[\"change_cfs\"] = df[\"cfs\"].diff().fillna(0.0)\n"
    "\n"
    "flow_file_path = \"09405500AZC_flow_clean.csv\"\n"
    "df.to_csv(flow_file_path)\n"
    "print(df.shape)\n"
    "df[[\"hour_updated\", \"cfs\", \"p01m\", \"tmpf\", \"dwpf\"]].tail()"
))

cells.append(nbf.v4.new_markdown_cell("## 3. Install Flow Forecast"))

cells.append(nbf.v4.new_code_cell(
    "os.chdir(\"flow-forecast\")\n"
    "!pip install -r requirements.txt\n"
    "!python setup.py develop\n"
    "os.chdir(\"..\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. (Optional) Weights & Biases\n"
    "\n"
    "Flow Forecast logs to W&B. Log in below, or set `\"wandb\": False` in the config to skip it."
))

cells.append(nbf.v4.new_code_cell(
    "# !wandb login\n"
    "# os.environ['MODEL_BUCKET'] = \"predict_cfs\"\n"
    "# os.environ[\"ENVIRONMENT_GCP\"] = \"Colab\"\n"
    "# os.environ[\"GCP_PROJECT\"] = \"gmap-997\""
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Build the NARX config\n"
    "\n"
    "Unlike the Informer (which needs the `TemporalLoader` and a decoder), NARX uses the plain "
    "`default` `CSVDataLoader`. Key `model_params`:\n"
    "\n"
    "- **`n_time_series`** — total input columns (targets + exogenous). Here 4: `cfs`, `p01m`, `tmpf`, `dwpf`.\n"
    "- **`n_targets`** — number of autoregressive targets. Must be the *first* columns of `relevant_cols`. Here 1 (`cfs`).\n"
    "- **`forecast_history`** — length of the input window.\n"
    "- **`output_seq_len`** — steps predicted per forward pass; set equal to `forecast_length`.\n"
    "- **`n_target_lags` / `n_exog_lags`** — how many past steps of the targets / exogenous inputs to feed the MLP (<= `forecast_history`).\n"
    "- **`hidden_size`, `num_hidden_layers`, `dropout`, `activation`** — MLP capacity."
))

cells.append(nbf.v4.new_code_cell(
    "FORECAST_HISTORY = 48\n"
    "FORECAST_LENGTH = 24\n"
    "\n"
    "\n"
    "def make_narx_config(flow_file_path, weight_path=None):\n"
    "    config = {\n"
    "        \"model_name\": \"NARX\",\n"
    "        \"model_type\": \"PyTorch\",\n"
    "        \"model_params\": {\n"
    "            \"n_time_series\": 4,\n"
    "            \"forecast_history\": FORECAST_HISTORY,\n"
    "            \"output_seq_len\": FORECAST_LENGTH,\n"
    "            \"n_targets\": 1,\n"
    "            \"n_target_lags\": FORECAST_HISTORY,\n"
    "            \"n_exog_lags\": FORECAST_HISTORY,\n"
    "            \"hidden_size\": 64,\n"
    "            \"num_hidden_layers\": 2,\n"
    "            \"dropout\": 0.1,\n"
    "            \"activation\": \"tanh\",\n"
    "        },\n"
    "        \"dataset_params\": {\n"
    "            \"class\": \"default\",\n"
    "            \"training_path\": flow_file_path,\n"
    "            \"validation_path\": flow_file_path,\n"
    "            \"test_path\": flow_file_path,\n"
    "            \"batch_size\": 64,\n"
    "            \"forecast_history\": FORECAST_HISTORY,\n"
    "            \"forecast_length\": FORECAST_LENGTH,\n"
    "            \"train_start\": 1000,\n"
    "            \"train_end\": 50000,\n"
    "            \"valid_start\": 50001,\n"
    "            \"valid_end\": 57000,\n"
    "            \"test_start\": 57000,\n"
    "            \"test_end\": 58000,\n"
    "            \"sort_column\": \"hour_updated\",\n"
    "            \"target_col\": [\"cfs\"],\n"
    "            \"relevant_cols\": [\"cfs\", \"p01m\", \"tmpf\", \"dwpf\"],\n"
    "            \"scaler\": \"StandardScaler\",\n"
    "            \"interpolate\": {\n"
    "                \"method\": \"back_forward_generic\",\n"
    "                \"params\": {\"relevant_columns\": [\"cfs\", \"p01m\", \"tmpf\", \"dwpf\"]},\n"
    "            },\n"
    "        },\n"
    "        \"training_params\": {\n"
    "            \"criterion\": \"MSE\",\n"
    "            \"optimizer\": \"Adam\",\n"
    "            \"optim_params\": {},\n"
    "            \"lr\": 0.001,\n"
    "            \"epochs\": 5,\n"
    "            \"batch_size\": 64,\n"
    "        },\n"
    "        \"early_stopping\": {\"patience\": 2},\n"
    "        \"GCS\": False,\n"
    "        \"wandb\": False,\n"
    "        \"forward_params\": {},\n"
    "        \"metrics\": [\"MSE\"],\n"
    "        \"inference_params\": {\n"
    "            \"datetime_start\": \"2020-05-31\",\n"
    "            \"hours_to_forecast\": 336,\n"
    "            \"test_csv_path\": flow_file_path,\n"
    "            \"decoder_params\": {\n"
    "                \"decoder_function\": \"simple_decode\",\n"
    "                \"unsqueeze_dim\": 1,\n"
    "            },\n"
    "            \"dataset_params\": {\n"
    "                \"file_path\": flow_file_path,\n"
    "                \"sort_column\": \"hour_updated\",\n"
    "                \"forecast_history\": FORECAST_HISTORY,\n"
    "                \"forecast_length\": FORECAST_LENGTH,\n"
    "                \"relevant_cols\": [\"cfs\", \"p01m\", \"tmpf\", \"dwpf\"],\n"
    "                \"target_col\": [\"cfs\"],\n"
    "                \"scaling\": \"StandardScaler\",\n"
    "                \"interpolate_param\": {\n"
    "                    \"method\": \"back_forward_generic\",\n"
    "                    \"params\": {\"relevant_columns\": [\"cfs\", \"p01m\", \"tmpf\", \"dwpf\"]},\n"
    "                },\n"
    "            },\n"
    "        },\n"
    "    }\n"
    "    if weight_path:\n"
    "        config[\"weight_path\"] = weight_path\n"
    "    return config"
))

cells.append(nbf.v4.new_markdown_cell("## 6. Train the NARX model"))

cells.append(nbf.v4.new_code_cell(
    "from flood_forecast.trainer import train_function\n"
    "\n"
    "# flow_file_path was set in the data-prep cell; we never left /content so it is still valid.\n"
    "config = make_narx_config(flow_file_path=flow_file_path)\n"
    "trained_model = train_function(\"PyTorch\", config)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. Inspect a single prediction\n"
    "\n"
    "Pull one `(x, y)` example from the test loader and run it through the model. NARX returns a "
    "tensor of shape `(batch, output_seq_len)` for a single target."
))

cells.append(nbf.v4.new_code_cell(
    "import torch\n"
    "\n"
    "x, y = trained_model.test_data[0]\n"
    "x_in = x.unsqueeze(0).to(trained_model.device).float()\n"
    "with torch.no_grad():\n"
    "    pred = trained_model.model(x_in).cpu()\n"
    "print(\"input shape :\", x_in.shape)\n"
    "print(\"output shape:\", pred.shape)\n"
    "print(\"scaled preds:\", pred[0])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. Multi-step closed-loop forecast & plot\n"
    "\n"
    "`infer_on_torch_model` runs the closed-loop `simple_decode` rollout (feeding predictions back in) "
    "and returns predictions alongside the history dataframe, which we plot against the observed flow."
))

cells.append(nbf.v4.new_code_cell(
    "from flood_forecast.evaluator import infer_on_torch_model\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "df_pred, end_tensor, hist, idx, test_loader, df_pred_samples = infer_on_torch_model(\n"
    "    trained_model, **config[\"inference_params\"]\n"
    ")\n"
    "\n"
    "plt.figure(figsize=(14, 5))\n"
    "plt.plot(df_pred[\"cfs\"].values, label=\"observed cfs\")\n"
    "plt.plot(df_pred[\"preds\"].values, label=\"NARX forecast\")\n"
    "plt.xlabel(\"hours\")\n"
    "plt.ylabel(\"cfs\")\n"
    "plt.title(\"Virgin River (09405500AZC) — NARX closed-loop forecast\")\n"
    "plt.legend()\n"
    "plt.show()"
))

nb["cells"] = cells
nb["metadata"] = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("NARX_Virgin_Predict.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote NARX_Virgin_Predict.ipynb")
