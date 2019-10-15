from collections import Counter

import numpy as np
import pandas as pd
import pickle

# Input parameters
import os, sys, getopt

from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True)

    for i in range(1, hm_days + 1):
        # Calculate percentage of change
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

''' 
Return buy/sell/hold based on prediction 
'''
def buy_sell_hold(*args):
    cols = [c for c in args]

    # % change in stock price would trigger the if statement
    req = 0.02

    for col in cols:
        if col > req:
            return"Buy"
            #return 1
        if col < -req:
            return "Sell"
            #return -1

    return "Hold"
    #return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)

    # Replace infinite change (Stock goes sideways) with np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace = True)

    # Normalized % change data for all data - X_training data
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    # Classification, the buy_sell_hold() function
    y = df['{}_target'.format(ticker)].values

    return X, y, df

def simple_classifier(ticker):
    X, y , df = extract_featuresets(ticker)

    # Split data into 75%:25% - training:testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)
    # % chance for accuracy, want > 33% from random selection
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    print('Confidence:', confidence)
    print('Predicted spread:', Counter(predictions))

    return confidence

def voting_classifier(ticker):
    X, y , df = extract_featuresets(ticker)

    # Split data into 75%:25% - training:testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Use different classifiers to predict and then averaged by the function VotingClassifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier(n_estimators=100))])

    clf.fit(X_train, y_train)
    # % chance for accuracy, want > 33% from random selection
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    print('Confidence:', confidence)
    print('Predicted spread:', Counter(predictions))

    return confidence



if __name__ == "__main__":
  import os, sys, getopt
  def usage():
       print ('Usage:	' + os.path.basename(__file__) + ' ticker ')
       print ('Options:')
       print ('\t--ticker=n (such as FB, SNAP, etc.)')
       sys.exit(2)
  # extract parameters
  try:
     opts, args = getopt.getopt(sys.argv[1:],"hedk:",["help", "ticker="])
  except getopt.GetoptError as err:
     print(err)
     usage()

  ticker = None
  filename = args[0] if len(args) > 0 else None
  for opt, arg in opts:
       if opt in ("-h", "--help"):
          usage()
       elif opt in ("-t", "--ticker"):
          ticker = arg
  # check arguments

  if (ticker is None):
      print('ticker symbol is missing\n')
      usage()
      sys.exit(1)

  voting_classifier(ticker)










