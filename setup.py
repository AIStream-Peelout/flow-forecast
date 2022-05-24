# flake8: noqa
from setuptools import setup
import os

library_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = f'{library_folder}/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

dev_requirements = [
    'autopep8',
    'flake8'
]

setup(
    name='flood_forecast',
    version='0.9988dev',
    packages=[
        'flood_forecast',
        'flood_forecast.transformer_xl',
        'flood_forecast.preprocessing',
        'flood_forecast.da_rnn',
        "flood_forecast.basic",
        "flood_forecast.meta_models",
        "flood_forecast.gcp_integration",
        "flood_forecast.deployment",
        "flood_forecast.custom"],
    license='GPL 3.0',
    description="An open source framework for deep time series forecasting and classfication built with PyTorch.",
    long_description='Flow Forecast is the top open source deep learning for time series forecasting and classification framework. We were the original TS framework to contain models like the transformer and have now expanded to include all popular deep learning models.',
    install_requires=install_requires,
    extras_require={
        'dev': dev_requirements})
