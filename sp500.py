# Web scraper
import bs4 as bs
import pickle
import requests

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

save_sp500_symbols()