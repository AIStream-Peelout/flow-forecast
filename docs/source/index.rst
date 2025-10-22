.. Flow Forecast documentation master file
   Created by sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Flow Forecast's documentation!
=========================================

Flow Forecast is a deep learning framework for time series forecasting written in PyTorch.
It makes it easy to train PyTorch Forecast models on a wide variety of datasets.
This documentation describes the internal Python code that makes up Flow Forecast.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules:

   evaluator
   long_train
   model_dict_function
   pre_dict
   pytorch_training
   time_model
   trainer
   explain_model_output
   utils
   basic_ae
   crossformer
   custom_types
   d_linear
   meta_models
   modules
   train_da
   training_utils
   utils_da


.. automodule:: flood_forecast.deployment
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Deployment:

   inference

.. automodule:: flood_forecast.preprocessing
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Preprocessing:

   interpolate_preprocess
   buil_dataset
   closest_station
   data_converter
   preprocess_da_rnn
   preprocess_metadata
   process_usgs
   pytorch_loaders
   temporal_feats

.. automodule:: flood_forecast.custom
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Custom:

   custom_opt

.. automodule:: flood_forecast.transformer_xl
   :members:

.. toctree::
   :maxdepth: 2
   :caption: TransformerXL:

   dummy_torch
   itransformer
   lower_upper_config
   multi_head_base
   transformer_basic
   transformer_xl
   transformer_bottleneck
   informer

.. automodule:: flood_forecast.gcp_integration
   :members:

.. toctree::
   :maxdepth: 3
   :caption: GCP Integration:

   basic_utils

.. automodule:: flood_forecast.da_rnn
   :members:

.. toctree::
   :maxdepth: 3
   :caption: DA RNN:

   model

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
