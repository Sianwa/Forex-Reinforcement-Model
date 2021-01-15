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
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
from oandapyV20.exceptions import V20Error
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

#------------------------------------------------------------dataset for training
def dataset_loader(currency_pair):
    forex_data = data_reader.DataReader(currency_pair,
                            "av-forex-daily",
                             api_key=api_key,
                             start='2010-1-1',
                             end='2015-12-31')
                             
    close = forex_data['close']

    return close
#---------------------------------------------------------------------load forex dataset for testing
def load_mydata(currency_pair):
    tweet_sentiments = pd.read_csv("data/EURUSD_Sentiments.csv",index_col=['date'])
    myData = data_reader.DataReader(currency_pair,
                                     "av-forex-daily",
                                     api_key=api_key,
                                     start='2015-1-1',
                                     end= '2015-3-31')

    close = myData['close']

    ohlc_data = close
    mData = pd.concat([ohlc_data,tweet_sentiments],axis=1,join='outer',sort=True)
    #to account for weekends we delete rows with null closing prices
    marketData = mData.dropna(axis=0,how='any')
    
    marketData.to_csv("marketdata.csv")
    
    sentimentData = []
    for i, rows in marketData.iterrows():
        my_list = [rows.close, rows.sentiment] 
        sentimentData.append(my_list)

    return sentimentData


#---------------------------------------------------the next function is for normalizing the data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#---------------------------------------------------finally we need a state creator that uses price difference 

def CreateState(data, timestep, window_size):
    #create look-back function
    starting_id = timestep - window_size + 1

    if starting_id >=0:
         windowData = data[starting_id:timestep + 1]
    else:    
         windowData = -starting_id * [data[0]] + list(data[0:timestep+1])

    state =[]
    for i in range(window_size- 1):
          state.append(windowData[i])
    return np.array([state])  
#---------------------------------------------------state creator that uses the closing price plus indicators.

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

        new_state = np.array(state)
    return new_state


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

def placeMrketOrder(currency_pair, lot_size):
    client_obj = oandapyV20.API(access_token = token)
    mktOrder = MarketOrderRequest( instrument = currency_pair,
                                    units = lot_size,
                                    ).data
    client_request = orders.OrderCreate(accountID = accountID, data = mktOrder)
    try: 
        rv = client_obj.request(client_request)
    except V20Error as err:
        print("ERROR OCCURED DUE TO : {}".format(err))
    else:        
        response = json.dumps(rv, indent = 2)
        data = json.loads(response)

        return  data["orderCreateTransaction"]["id"]

#-----------------------------------------------------------close specific position

def closePosition(trade_id):
    client_obj = oandapyV20.API(access_token = token)
    client_request = trades.TradeClose(accountID = accountID, tradeID = trade_id)

    try:
        rv = client_obj.request(client_request)
    except V20Error as err:
        print("ERROR OCCURED DUE TO : {}".format(err))
    else:
        response = json.dumps(rv, indent=2)
        return response

#-----------------------------------------------------------------close all open positions        

def killSwitch(currency_pair):
    client = oandapyV20.API(access_token = token)
    data = { "longUnits" : "ALL"}

    client_request = positions.PositionClose(accountID = accountID,
                                            instrument= currency_pair,
                                            data = data)
    try:
        rv = client.request(client_request)
    except V20Error as err:
        print("ERROR OCCURED DUE TO : {}".format(err))
    else:
        response = json.dumps(rv, indent = 2)
        return response

#--------------------------------------------------------------plot reward per episode
def plot_totalReward(dataframe):
    title = "Model's Performance: Reward per Episode"
    plt.figure(figsize=(12,5))
    plt.scatter(dataframe.index, dataframe['Reward'], color='blue',alpha=1)

    plt.title(title)
    plt.xlabel('Episode',fontsize=12)
    plt.ylabel('Rewards:Profit/Loss',fontsize=12)

    return plt.show()