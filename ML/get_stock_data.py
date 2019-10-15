# Web scraper
import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from pandas_datareader._utils import RemoteDataError

style.use('ggplot')

"""
    Function specifically created to save the data for the SP500 symbols from Wikipedia
    
    Takes about 10 minutes for it to run and finish, because it checks if yahoo has data for
    each ticker found on Wikipedia 
"""
def save_sp500_tickers():
    # Request wikipedia for sp500 companies list
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []

    # For each ticker symbol, ping yahoo API for data, if exception raised, don't add it into sp500 array
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.strip()

        start = dt.datetime.today() - dt.timedelta(days=1)
        end = dt.datetime.today()

        # Ping yahoo
        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
            tickers.append(ticker)
            print(ticker, " appended to sp500 list")
        except:
            print(ticker, " not found with Yahoo finance data")

    # Write tickers array into pickle file
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers

'''
    Stocks helper function for future use in specific stocks
'''
def dump_own_stocks():
    symb = ["BND", "VXUS", "ACB", "FB", "SNAP"]
    with open("myStocks.pickle", "wb") as f:
        pickle.dump(symb, f)

'''
    Function to import Yahoo finance data into individual CSV files 
'''
def get_data_yahoo(reload_stocks = False):
    # Check reload_stocks parameters
    if reload_stocks:
        save_sp500_tickers()

    with open("sp500tickers.pickle", "rb") as f:
        symbols = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2013,1,1)
    end = dt.datetime.today()

    # For each symbol save their stock info in a the directory 'stocks_dfs'
    for ticker in symbols:
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                if (df is None):
                    raise Exception

                df.to_csv('stocks_dfs/{}.csv'.format(ticker))
            except:
                print(ticker, " not found")
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        symbols = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(symbols):
        try:
            df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
        except:
            print('stocks_dfs/{}.csv'.format(ticker))

        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close' : ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')

    #df['BND'].plot()
    #plt.show()

    # Generate correlation table
    df_corr = df.corr()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor = False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor = False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.title("Correlation Matrix between Stocks")

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

save_sp500_tickers()
get_data_yahoo()
compile_data()

