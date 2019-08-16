from setuptools import setup
from setuptools import find_packages


setup(
    name='flood_forecast',
    version='0.01dev',
    packages=['flood_forecast', 'flood_forecast.api_connectors', 'flood_forecast.preprocessing', 'flood_forecast.da_rnn'],
    license='Public',
    long_description='A public package  ',
    install_requires=['scikit', 'torch', 'tensorflow', 'pandas"]
)
