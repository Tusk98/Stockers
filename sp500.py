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

style.use('ggplot')

"""
    Function specifically created to save the data for the SP500 symbols from Wikipedia
    Doesn't seem to work
"""
def save_sp500_symbols():
    # Make a request from the wikipedia page
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    symbols = []

    # Take the elements in column [0] since those are the symbols
    for row in table.findAll('tr')[1:]:
        sym = row.findAll('td')[0].text
        symbols.append(sym)

    # Dump into sp500symbols.pickle
    with open("sp500symbols.pickle", "wb") as f:
        pickle.dump(symbols, f)

    return symbols

def dump_own_stocks():
    symb = ["BND", "VXUS", "ACB", "FB", "SNAP"]
    with open("myStocks.pickle", "wb") as f:
        pickle.dump(symb, f)


def get_data_yahoo(reload_stocks = False):
    # Check if myStocks file exist
    symbols = ["BND", "VXUS", "ACB", "FB", "SNAP"]
    if reload_stocks:
        dump_own_stocks()

    with open("myStocks.pickle", "rb") as f:
        symbols = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2013,1,1)
    end = dt.datetime.today()

    # For each symbol save their stock info in a the directory stocks_dfs
    for ticker in symbols:
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stocks_dfs/{}.csv'.format(ticker))
            except:
                print(ticker, " not found")
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("myStocks.pickle", "rb") as f:
        symbols = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(symbols):
        df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
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
    main_df.to_csv('myStocks_joined_closes.csv')

def visualize_data():
    df = pd.read_csv('myStocks_joined_closes.csv')

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

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

get_data_yahoo(True)
compile_data()
visualize_data()