# Deep learning for time series forecasting
This repository is an open-source DL for time series library. Currently [Task Time Series from CoronaWhy](https://github.com/CoronaWhy/task-ts/wiki) maintains this repo. Pull requests are welcome. Historically, this repository provided open source benchmark and codes for flash flood and river flow forecasting. 

| branch  | status                                                                                                                                                                                                            |
| ---     | ---                                                                                                                                                                                                               |
| master  | [![CircleCI](https://circleci.com/gh/AIStream-Peelout/flow-forecast.svg?style=svg&circle-token=f7be0a4863710165969ba0903fa471f08a347df1)](https://circleci.com/gh/AIStream-Peelout/flow-forecast)                 |
| Build PY| ![Upload Python Package](https://github.com/AIStream-Peelout/flow-forecast/workflows/Upload%20Python%20Package/badge.svg)|
| Docs | [Link](https://flow-forecast.readthedocs.io/en/readthedocs/)|
## Getting Started 

Using the library:
1. Run `pip install flood-forecast`
2. See the training models page in the [Wiki](https://github.com/AIStream-Peelout/flow-forecast/wiki/Training-models).
3. Deployment is still a work in progress but high priority in the road-map.

**Models currently supported**

1. Vanilla LSTM: A basic multivariate LSTM with an optional embedding layer. 
2. Full transformer: The full transformer model (based on the PyTorch implementation). This requires using a specialized decoder and supplying the target.  
3. Simple Multi-Head Attention: A single multi-head attention mechanism followed by linear embedding layers. Suitable for transfer learning of uneven shapes. 
4. Transformer w/ a linear decoder: Linear embedding layers followed by n-stacks of the transformer blocks. Suitable for transfer learning of uneven shapes.
5. DA-RNN (CPU only for now): A good all around time series forecasting model.

**Integrations**

[Google Cloud Platform]() 

[Weights and Biases]() 

Dataverse

## Contributing 

For instructions on contributing please see our [contributions page](http://github.com/AIStream-Peelout/flow-forecast/wiki/Contribution-Guidelines) and our [project board](https://github.com/AIStream-Peelout/flow-forecast/projects). 


## Historical Tasks 

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
 
