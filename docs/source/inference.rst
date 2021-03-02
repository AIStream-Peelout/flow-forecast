Inference
=========================

This API makes it easy to run inference on trained PyTorchForecast modules. To use this code you 
need three main files: your model's configuration file, a CSV containing your data, and a path to 
your model weights.

.. code-block:: python
   :caption: example initialization
   
   import json
   from datetime import datetime
   from flood_forecast.deployment.inference import InferenceMode
   new_water_data_path = "gs://predict_cfs/day_addition/01046000KGNR_flow.csv"
   weight_path = "gs://predict_cfs/experiments/10_December_202009_34PM_model.pth"
   with open("config.json") as y:
     config_test = json.load(y)
   infer_model = InferenceMode(336, 30, config_test, new_water_data_path, weight_path, "river")

.. code-block:: python 
    :caption: example plotting

.. automodule:: flood_forecast.deployment.inference
    :members:
