from setuptools import setup

setup(
    name='flood_forecast',
    version='0.01dev',
    packages=['flood_forecast', 'flood_forecast.transformer_xl', 'flood_forecast.preprocessing', 'flood_forecast.da_rnn', "flood_forecast.basic", "flood_forecast.custom"],
    license='Public',
    long_description='A public package for forecasting river flows and flash flood severity',
    install_requires=['scikit-learn', 'torch', 'tensorflow', 'pandas', 'google-cloud','sphinx','sphinx-rtd-theme','sphinx-autodoc-typehints']
)
