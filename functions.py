import pandas_datareader.data as data_reader
import datetime as dt
import numpy as np
import math
import json
import yaml
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    TakeProfitDetails,
    StopLossDetails)
plt.style.use('fivethirtyeight')


#--------------------------------------------------------------------load necessary API tokens to access data
with open('AccConfig.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    api_key = data.get("alpha_vantage_api")
    token = data.get("token")
    accountID = data.get("account")    

#----------------------------------------------------------------------this is a formating function
# to print out the prices of the buy and sell trades that were executed.
def formatPrice(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))    

#this next function connects with a data source and pulls the data from it. For now we have a csv so we wont use OANDA API.
#another option is pandas_datareader that gives remote access to data


#---------------------------------------------------------------------load forex dataset with SMA and MACD indicator
def forexdata_loader(currency_pair):
    forex_data = data_reader.DataReader(currency_pair,
                            "av-forex-daily",
                             api_key=api_key,
                             start='2010-1-1',
                             end='2010-12-31')

    close = forex_data['close']

    #next we want to add the indicators to our dataset
    df = close
    sma = ta.SMA(close, timeperiod=14)
    sma_df = pd.DataFrame(sma, columns=['SMA'])
    macd, signal, hist = ta.MACD(close, fastperiod=12,slowperiod=26,signalperiod=9)
    macd_df = pd.DataFrame(macd, columns=['MACD'])
    signal_df = pd.DataFrame(signal,columns=['SIGNAL'])
    hist_df = pd.DataFrame(hist, columns=['HIST'])
    final = pd.concat([df,sma_df,macd_df,signal_df], axis=1,join='outer', ignore_index=True)

    #drop null values and return final usable data
    final_data = final.dropna()
    final_data.to_csv("data/training.csv")

    return final_data[0]


#---------------------------------------------------the next function is for normalizing the data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#---------------------------------------------------finally we need a state creater function that takes the data and generates states from it

def state_creator(data, timestep, window_size):

    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep + 1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0:timestep + 1])

    state = []
    #for the state we want it to be the price difference between data at consecutive timestep
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])


#-------------------------------------------------------plotting function
def plotData(dataSet):
    title = "Model's performance"
    plt.figure(figsize=(12.2,4.5))
    plt.scatter(dataSet.index, dataSet['Buy'], color = 'green', label='Buy Signal', marker = '^', alpha = 1)
    plt.scatter(dataSet.index, dataSet['Sell'], color = 'red', label='Sell Signal', marker = 'v', alpha = 1)
    plt.plot( dataSet['Data'],  label='Tick Data',alpha = 0.35)
       
    plt.title(title)
    plt.xlabel('Dates',fontsize=12)
    plt.ylabel('Price Value',fontsize=12)
    plt.legend(dataSet.columns.values, loc='upper left')

    return plt.show()
#

#------------------------------------------------------oanda function to return account details

def getAccountDetails():
    client = oandapyV20.API(access_token = token)
    client_request = accounts.AccountSummary(accountID = accountID)
    rv = client.request(client_request)
    response = json.dumps(rv)

    data = json.loads(response)

    return data['account']['balance']

#-------------------------------------------------------oanda function to place market order

def placeMrketOrder():
    client_obj = oandapyV20.API(access_token = token)
    mktOrder = MarketOrderRequest( instrument = "GBP_USD",
                                    units = 1000,
                                    ).data
    client_request = orders.OrderCreate(accountID = accountID, data = mktOrder)
    try: 
        rv = client_obj.request(client_request)
    except V20Error as err:
        print("ERROR OCCURED DUE TO : {}".format(err))
    else:        
        response = json.dumps(rv, indent = 2)
        return  response