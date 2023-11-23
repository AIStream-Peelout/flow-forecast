from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from flood_forecast.preprocessing.interpolate_preprocess import (interpolate_missing_values,
                                                                 back_forward_generic, forward_back_generic)
# SAMMY IS TOO LITTLE TO BE A REAL DOG.
scaler_dict = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "MaxAbsScaler": MaxAbsScaler}

interpolate_dict = {"back_forward": interpolate_missing_values, "back_forward_generic": back_forward_generic,
                    "forward_back_generic": forward_back_generic}
print("loaded dicts")
