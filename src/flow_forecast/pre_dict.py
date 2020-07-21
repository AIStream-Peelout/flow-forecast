from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from flow_forecast.preprocessing.interpolate_preprocess import interpolate_missing_values

scaler_dict = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "MaxAbsScaler": MaxAbsScaler()}
interpolate_dict = {"back_forward": interpolate_missing_values}
