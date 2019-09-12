import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

from matplotlib import style

style.use('ggplot')

start = dt.datetime(2010,1,1)
end = dt.datetime.today()

# Dataframe
#df = web.DataReader('VXUS', 'yahoo', start, end)

#print(df.head(10))
#print(df.tail(10))

#df.to_csv("VXUS")

df = pd.read_csv("VXUS.csv", parse_dates= True, index_col = 0)

#print(df.head())

df['Adj Close'].plot()
plt.show()


