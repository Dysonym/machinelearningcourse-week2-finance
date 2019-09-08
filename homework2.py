import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import isfile
import math

from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline




#Getting the Data
if not isfile("AMD.csv"):
    import yfinance as yf
    from pandas_datareader import data as pdr
    print("Downloading AMD data from yahoo finance")
    yf.pdr_override()
    df_full = pdr.get_data_yahoo("AMD", start="2018-01-01").reset_index()
    df_full.to_csv('AMD.csv',index=False)


amd = pd.read_csv('AMD.csv')
amd.set_index(["Date"], inplace=True)
forecast_col = 'Adj Close'


#https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

amd["HighLow_PCT"] = (amd['High'] - amd['Low']) / amd['Close'] * 100.0
amd["DayPCT_Change"] = (amd['Close'] - amd['Open']) / amd['Open'] * 100.0

#Extra features -- adding some context
amd["Close Yesterday"] = amd[forecast_col].shift(1)
amd["Volume Yesterday"] = amd["Volume"].shift(1)
amd["Last3"] = amd[forecast_col].rolling(window=3).mean() # average of the last 3 days
amd["Last5"] = amd[forecast_col].rolling(window=5).mean() # average of the last 5 days
amd.drop(amd.index[0:5], inplace=True) # drop rows which don't have the moving average

#forecast_out = int(math.ceil(0.01 * len(amd))) # == 5
# I think predicting tomorow is good enough, 
# why try to predict 5 days in the future just from today's data

forcast = 1 # used to shift data to get y
lately = 10 # Days of Data to Keep for testing


amd['label'] = amd[forecast_col].shift(forcast*-1) #predict tomorow

# X
X = np.array(amd.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
X_lately = X[-lately:]
X = X[:-lately]

y = np.array(amd['label'])
y = y[:-lately]

print('Dimension of X',X.shape)
print('Dimension of y',y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)

models = [
    {
        "name": "Linear",
        "model" : LinearRegression(n_jobs=-1)
    },
    {
        "name": "Quad2",
        "model": make_pipeline(PolynomialFeatures(2), Ridge())
    },
    {
        "name": "Quad3",
        "model": make_pipeline(PolynomialFeatures(3), Ridge())
    },
    {
        "name": "KNN",
        "model": KNeighborsRegressor(n_neighbors=2)
    }

]

#fit models and get confidence score
for m in models:
    #print("fit: ",m["name"])
    m["model"].fit(X_train, y_train)
    m["score"] = m["model"].score(X_test, y_test)
    print("Model {} confidence score: {}".format(m["name"], m["score"]) )

for i, m in enumerate(models[:4]):
    forcast_label = "Forecast-" + m["name"]
    forecast_set = m["model"].predict(X_lately)
    #amd.loc["Forecast " + m["name"]].iloc[-lately:] = forecast_set
    amd[forcast_label] = np.nan
    amd.loc[:, forcast_label].iloc[-lately:] = forecast_set

    plt.figure(i)
    plt.suptitle(m["name"])
    amd["label"].tail(10).plot(marker=".")
    amd[forcast_label].tail(10).plot(marker=".")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc=4)


print(amd.drop(["Open", "High", "Low", "Close", "Volume", "HighLow_PCT", "DayPCT_Change"], 1).iloc[-lately:])
#plt.show()
print("Will now try the QLearning Agent from the Q&A CodeLab ")


close = amd["Adj Close"].values.tolist()

from QLearningAgent import Agent as QAgent

initial_money = 10000
window_size = 30
skip = 1
batch_size = 32
agent = QAgent(state_size = window_size, 
              window_size = window_size, 
              trend = close, 
              skip = skip, 
              batch_size = batch_size)
agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

states_buy, states_sell, total_gains, invest = agent.buy(initial_money, close)


plt.figure(i+1)

plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.suptitle('QLearning Agent: total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
plt.show()

# Maybe we could plugin the KNN result into the Qlearning agent trend param?
# Too tired, going to sleep now ...


    


##import pdb; pdb.set_trace()







