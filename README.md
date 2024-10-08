# Deep learning for time series forecasting, classification, and anomaly detection
![Example image](https://raw.githubusercontent.com/CoronaWhy/task-ts/master/images/Picture1.png)
Flow Forecast (FF) is an open-source deep learning for time series forecasting framework. It provides all the latest state of the art models (transformers, attention models, GRUs, ODEs) and cutting edge concepts with easy to understand interpretability metrics, cloud provider integration, and model serving capabilities. Flow Forecast was the first time series framework to feature support for transformer based models and remains the only true end-to-end deep learning for time series framework. Currently, [Task-TS from CoronaWhy](https://github.com/CoronaWhy/task-ts/wiki) primarily maintains this repository. Pull requests are welcome. Historically, this repository provided open source benchmark and codes for flash flood and river flow forecasting.

For additional tutorials and examples please see our [tutorials repository](https://github.com/AIStream-Peelout/flow_tutorials).

| branch  | status                                                                                                                    $$                                                                                        |
| ---     | ---                                                                                                                                                                                                               |
| master  | [![CircleCI](https://dl.circleci.com/status-badge/img/gh/AIStream-Peelout/flow-forecast/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/AIStream-Peelout/flow-forecast/tree/master)                |              |
| Build PY| ![Upload Python Package](https://github.com/AIStream-Peelout/flow-forecast/workflows/Upload%20Python%20Package/badge.svg)|
| Documentation | [![Documentation Status](https://readthedocs.org/projects/flow-forecast/badge/?version=latest)](https://flow-forecast.readthedocs.io/en/latest/)|
| CodeCov| [![codecov](https://codecov.io/gh/AIStream-Peelout/flow-forecast/branch/master/graph/badge.svg)](https://codecov.io/gh/AIStream-Peelout/flow-forecast)|
| CodeFactor| [![CodeFactor](https://www.codefactor.io/repository/github/aistream-peelout/flow-forecast/badge)](https://www.codefactor.io/repository/github/aistream-peelout/flow-forecast)|
## Getting Started

Using the library
1. Run `pip install flood-forecast`
2. Detailed info on training models can be found on the [Wiki](https://flow-forecast.atlassian.net/wiki/spaces/FF/pages/364019713/Training+Models).
3. Check out our [Confluence Documentation](https://flow-forecast.atlassian.net/wiki/spaces/FF/overview)

**Models currently supported**

1. Vanilla LSTM (LSTM): A basic LSTM that is suitable for multivariate time series forecasting and transfer learning.
2. Full transformer (SimpleTransformer in model_dict): The full original transformer with all 8 encoder and decoder blocks. Requires passing the target in at inference.
3. Simple Multi-Head Attention (MultiHeadSimple): A simple multi-head attention block and linear embedding layers. Suitable for transfer learning.
4. Transformer with a linear decoder (CustomTransformerDecoder in model_dict): A transformer with n-encoder blocks (this is tunable) and a linear decoder. Suitable for forecasting, classification or anomaly detection.
5. [DA-RNN](https://arxiv.org/abs/1704.02971): (DARNN) A well rounded model with which utilizes a LSTM + attention.
6. [Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/abs/1907.00235) (called DecoderTransformer in model_dict):
7. [Transformer XL](https://arxiv.org/abs/1901.02860): Porting Transformer XL for time series.
8. [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) (Informer)
9. [DeepAR](https://arxiv.org/abs/1704.04110)
10. [DSANet](https://kyonhuang.top/files/DSANet/Huang-DSANet.pdf): Model from the DSANet paper with additional optional parameters.
11. Simple Linear Model (SimpleLinearModel): Essentially a linear regression model with a few more layers. Great for rapid prototyping of features and interpretability.
12.  Vanilla GRU with optional probablistic output layer. Good for multivariate time series forecasting and classification.
13. DLinear and NLinear from the AAAI paper [Are Transformers Effective for Time Series Forecasting](https://arxiv.org/abs/2205.13504)
14. [Crossformer](https://openreview.net/forum?id=vSVLM2j9eie) from ICLR 2023
15. [Anomaly Transformer](https://arxiv.org/abs/2110.02642)
16. [TSMixer](https://arxiv.org/abs/2303.06053)
17. [Inverted Transformer](https://arxiv.org/abs/2310.06625) + Various Attention Mechanisms

**Forthcoming Models**

We have a number of models we are planning on releasing soon. [Please check our project board for more info](https://github.com/AIStream-Peelout/flow-forecast/projects/5)

**Integrations**

[Google Cloud Platform](https://github.com/AIStream-Peelout/flow-forecast/wiki/Cloud-Provider-Integration)

[Weights and Biases](https://www.wandb.com/)

## Contributing

For instructions on contributing please see our [contributions page](https://flow-forecast.atlassian.net/wiki/spaces/FF/pages/11403276/Contributing) and our [project board](https://github.com/AIStream-Peeloutt/flow-forecast/projects/5).


## Historical River Flow Data

### Task 1 Stream Flow Forecasting
This task focuses on forecasting a stream's future flow/height (in either cfs or feet respectively) given factors such as current flow, temperature, and precipitation. In the future we plan on adding more variables that help with the stream flow prediction such as snow pack data and the surrounding soil moisture index.

### Task 2 Flood severity forecasting
Task two focuses on predicting the severity of the flood based on the flood forecast, population information, and topography. Flood severity is defined based on several factors including the number of injuires, property damage, and crop damage.

If you use either the data or code from this repository please use the citation below. Additionally please cite the original authors of the models.
```
@misc{godfried2020flowdb,
      title={FlowDB a large scale precipitation, river, and flash flood dataset},
      author={Isaac Godfried and Kriti Mahajan and Maggie Wang and Kevin Li and Pranjalya Tiwari},
      year={2020},
      eprint={2012.11154},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
