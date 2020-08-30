# Deep learning for time series forecasting
This repository is an open-source DL for time series library. Currently [Task-TS from CoronaWhy](https://github.com/CoronaWhy/task-ts/wiki) maintains this repo. Pull requests are welcome. Historically, this repository provided open source benchmark and codes for flash flood and river flow forecasting. 

| branch  | status                                                                                                                                                                                                            |
| ---     | ---                                                                                                                                                                                                               |
| master  | [![CircleCI](https://circleci.com/gh/AIStream-Peelout/flow-forecast.svg?style=svg&circle-token=f7be0a4863710165969ba0903fa471f08a347df1)](https://circleci.com/gh/AIStream-Peelout/flow-forecast)                 |
| Build PY| ![Upload Python Package](https://github.com/AIStream-Peelout/flow-forecast/workflows/Upload%20Python%20Package/badge.svg)|
| Documentation | [![Documentation Status](https://readthedocs.org/projects/flow-forecast/badge/?version=readthedocs)](https://flow-forecast.readthedocs.io/en/readthedocs/?badge=readthedocs)|
## Getting Started 

Using the library
1. Run `pip install flood-forecast`
2. Detailed info on training models can be found on the [Wiki](https://github.com/AIStream-Peelout/flow-forecast/wiki/Training-models).

**Models currently supported**

1. Vanilla LSTM: A basic LSTM that is suitable for multivariate time series forecasting and transfer learning. 
2. Full transformer: The full transformer with all 8 encoder and decoder blocks. Requires passing the target in at inference. 
3. Simple Multi-Head Attention: A simple multi-head attention block/embedding layers. Suitable for transfer learning.
4. Transformer w/a linear decoder: A transformer with n-encoder blocks (this is tunable) and linear decoder. Suitable for transfer learning.
5. DA-RNN (CPU only for now): A well rounded model with which utilizes a LSTM + attention. 

**Forthcoming Models**

We have a number of models we are planning on releasing soon. [Please check our project board](https://github.com/AIStream-Peelout/flow-forecast/projects/5)

**Integrations**

[Google Cloud Platform]() 

[Weights and Biases](https://www.wandb.com/)

## Contributing 

For instructions on contributing please see our [contributions page](http://github.com/AIStream-Peelout/flow-forecast/wiki/Contribution-Guidelines) and our [project board](https://github.com/AIStream-Peelout/flow-forecast/projects). 


## Historical River Flow Data  

### Task 1 Stream Flow Forecasting 
This task focuses on forecasting a stream's future flow/height (in either cfs or feet respectively) given factors such as current flow, temperature, and precipitation. In the future we plan on adding more variables that help with the stream flow prediction such as snow pack data and the surrounding soil moisture index. 

### Task 2 Flood severity forecasting
Task two focuses on predicting the severity of the flood based on the flood forecast, population information, and topography. Flood severity is defined based on several factors including the number of injuires, property damage, and crop damage.

If you use either the data or code from this repository please cite as
```
@inproceedings{GodfriedFlow2019,
Author = {Isaac Godfried},
Title = {Flow: A large scale dataset for stream flow and flood damage forecasting},
Booktitle  = {Arxiv Preprint},
Year = {2019}
}
```
 
