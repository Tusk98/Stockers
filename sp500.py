# Web scraper
import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

"""
    Function specifically created to save the data for the SP500 symbols from Wikipedia
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


def get_data_yahoo(reload_sp500 = False):
    if reload_sp500:
        symbols = save_sp500_symbols()
    else:
        with open("sp500symbols.pickle", "wb") as f:
            symbols = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2013,1,1)
    end = dt.datetime.today()

    for ticker in symbols:
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stocks_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

            