# Flow and flash flood forecasting benchmark
This repository provides open source benchmark and codes for flash flood and river flow forecasting. Specifically, it contains baseline methods for forecasting stream flows around the United States. Task two focuses on forecasting the severity of the flood based on the forecast and the surrounding area. Additionally, some of the pre-processing scripts used to form the dataset are included for reproduciblity purposes and to facilitate appending new gages and weather stations.

[![CircleCI](https://circleci.com/gh/isaacmg/flow-forecast.svg?style=svg&circle-token=f7be0a4863710165969ba0903fa471f08a347df1)](https://circleci.com/gh/isaacmg/flow-forecast)

Using the library
1. Run `pip install flood-forecast`
2. Download the data into the `data` folder

## Task 1 Stream Flow Forecasting 
This task focuses on forecasting a stream's future flow/height (in either cfs or feet respectively) given factors such as current flow, temperature, and precipitation. In the future we plan on adding more variables that help with the stream flow prediction such as snow pack data and the surrounding soil moisture index. 

## Task 2 Flood severity forecasting
Task two focuses on predicting the severity of the flood based on the flood forecast, population information, and topography. Flood severity is defined based on several factors including the number of injuires, property damage, and crop damage.

If you use either the data or codes from this repository please cite as
```
@inproceedings{GodfriedFlow2019,
Author = {Isaac Godfried},
Title = {Flow: A large scale dataset for stream flow and flood damage forecasting},
Booktitle  = {Arxiv Preprint},
Year = {2019}
}
```
