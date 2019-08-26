from setuptools import setup
from setuptools import find_packages


setup(
    name='flood_forecast',
    version='0.01dev',
    packages=['flood_forecast', 'flood_forecast.transformer_xl', 'flood_forecast.preprocessing', 'flood_forecast.da_rnn', "flood_forecast.basic"],
    license='Public',
    long_description='A public package for forecasting river flows and flash floods',
    install_requires=['scikit-learn', 'torch', 'tensorflow', 'pandas']
)
