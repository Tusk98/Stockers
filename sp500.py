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

def dump_own_stocks():
    symb = ["BND", "VXUS", "ACB", "FB"]
    with open("myStocks.pickle", "wb") as f:
        pickle.dump(symb, f)


def get_data_yahoo(reload_stocks = False):
    # Check if myStocks file exist
    if reload_stocks:
        symbols = dump_own_stocks()
    else:
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

compile_data()