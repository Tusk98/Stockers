# Stockers

This is a project meant for using some basic machine learning / data analysis on stock market data. 


main.py has a POC for graphing stock price change over time for the stock VXUS
In the folder that contains main.py and VXUS.csv, run

```
~ $ python3 main.py
```

The file get_stock_data.py loads the SP500 list of companies from Wikipedia and then requests Yahoo for stock information
from 2013 to whatever the date is when you run the program. Because of updating the SP500 list and their
stock prices, it will take at least several minutes to finish running the program.

After gathering stock data it will present a correlation matrix of the SP500 companies

```
~ $ python3 get_stock_data
```

## Dependencies
 - Python 3.X
   - [beatifulsoup4](https://pypi.org/project/beautifulsoup4/)
   - [matplotlib](https://pypi.org/project/matplotlib/)
   - [numpy](https://pypi.org/project/numpy/)
   - [pandas](https://pandas.pydata.org/)

## Features
