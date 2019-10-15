# Stockers

This is a project meant for using some basic machine learning / data analysis on stock market data. 

## Dependencies
 - Python 3.X
   - [beatifulsoup4](https://pypi.org/project/beautifulsoup4/)
   - [matplotlib](https://pypi.org/project/matplotlib/)
   - [numpy](https://pypi.org/project/numpy/)
   - [pandas](https://pandas.pydata.org/)
   - [sklearn](https://scikit-learn.org/stable/install.html)


main.py has a POC for graphing stock price change over time for the stock VXUS
In the folder that contains main.py and VXUS.csv, run

```
~ $ python3 main.py
```

The file **get_stock_data.py** loads the SP500 list of companies from Wikipedia and then requests Yahoo for stock information
from 2013 to whatever the date is when you run the program. Because of updating the SP500 list and their
stock prices, it will take at least several minutes to finish running the program.

After gathering stock data it will present a correlation matrix of the SP500 companies

```
~ $ python3 get_stock_data.py
```

The file **ml_classifiers.py** currently contains a basic ML Voting Classifier based on Linear SVC, K-Neighbors, and
Random Forest classifiers. The file takes the data created get_stock_data to train the model using a split of
75:25 - training:testing. It currently returns a number for Data Spread, confidence %, and predicted spread.
Only currently supports data imported by get_stock_data (i.e. SP500 companies)

```
~ $ python3 ml_classifiers.py -h
```
Will return a help message to get you started.


## Features
