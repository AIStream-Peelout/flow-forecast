.. Flow Forecast documentation master file, created by
   sphinx-quickstart on Fri Jun 26 17:56:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Flow Forecast's documentation!
=========================================
.. automodule:: flow_forecast

.. toctree::
   :maxdepth: 2
   :caption: General:

   utils
   evaluator
   long_train
   model_dict_function
   plot_functions
   pre_dict
   pytorch_training
   time_model
   trainer
   training_utils
   license
   help

.. automodule:: flow_forecast.preprocessing

.. toctree::
   :maxdepth: 3
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

.. automodule:: flow_forecast.custom

.. toctree::
   :maxdepth: 2
   :caption: Custom:

   custom_opt

.. automodule:: flow_forecast.transformer_xl

.. toctree::
  :maxdepth: 2
  :caption: TransformerXL:

  dummy_torch
  lower_upper_config
  multi_head_base
  transformer_basic
  transformer_xl



.. automodule:: flow_forecast.gcp_integration

.. toctree::
  :maxdepth: 3
  :caption: GCP Integration:

  basic_utils




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
