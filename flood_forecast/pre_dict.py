from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from flood_forecast.preprocessing.interpolate_preprocess import (interpolate_missing_values,
                                                                 back_forward_generic, forward_back_generic)
from torch import nn

scaler_dict = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler}

interpolate_dict = {"back_forward": interpolate_missing_values, "back_forward_generic": back_forward_generic,
                    "forward_back_generic": forward_back_generic}

final_layer_dict = {"relu": nn.relu, "softplus": nn.softplus, "sigmoid": nn.sigmoid}
