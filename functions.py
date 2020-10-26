import pandas_datareader.data as data_reader
import datetime as dt
import numpy as np
import math
import yaml
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#load necessary API tokens to access data
with open('AccConfig.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    api_key = data.get("alpha_vantage_api")

#this is a formating function to print out the prices of the buy and sell trades that were executed.
def formatPrice(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))    

#this next function connects with a data source and pulls the data from it. For now we have a csv so we wont use OANDA API.
#another option is pandas_datareader that gives remote access to data

#dataset for training
def dataset_loader(currency_pair):
    dataset = data_reader.DataReader(currency_pair,
                                     start='2015-1-1',
                                     end= '2015-2-1',
                                     data_source='yahoo')

    close = dataset['Close']
   
    return close

#load forex dataset with SMA and MACD indicator
def forexdata_loader(currency_pair):
    forex_data = data_reader.DataReader(currency_pair,
                            "av-forex-daily",
                             api_key=api_key,
                             start='2010-1-1',
                             end='2011-12-31')

    close = forex_data['close']

    #next we want to add the indicators to our dataset
    df = pd.DataFrame(close)
    sma = ta.SMA(close, timeperiod=14)
    sma_df = pd.DataFrame(sma)
    macd, signal, hist = ta.MACD(close, fastperiod=12,slowperiod=26,signalperiod=9)
    macd_df = pd.DataFrame(macd)
    signal_df = pd.DataFrame(signal)
    hist_df = pd.DataFrame(hist)
    final_data = pd.concat([df,sma_df,macd_df,signal_df,hist_df], axis=1,join='outer', ignore_index=True)

    return final_data


#dataset for testing 
def test_data_loader(currency_pair):
    data = data_reader.DataReader(currency_pair,
                                  start='2020-1-1',
                                  end= '2020-9-1',
                                  data_source='yahoo')    
    closing_price = data['Close']
    return closing_price

#the next function is the sigmoid activation function used for normalizing the data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#finally we need a state creater function that takes the data and generates states from it

def state_creator(data, timestep, window_size):

    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep + 1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0:timestep + 1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])

#plotting function
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



