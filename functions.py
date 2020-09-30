import pandas_datareader.data as data_reader
import datetime as dt
import numpy as np
import math

#this is a formating function to print out the prices
#of the buy and sell trades that were executed.

def formatPrice(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))    

#this next function connects with a data source and pulls the 
# data from it. For now we have a csv so we wont use OANDA API.
# another option is pandas_datareader that gives remote access to data

def dataset_loader(currency_pair):
    dataset = data_reader.DataReader(currency_pair,
                                     start='2015-1-1',
                                     end= '2015-2-1',
                                     data_source='yahoo')

    close = dataset['Close']
   
    return close
#the next function is the sigmoid activation function used for normalizing the
#data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#finally we need a state creater function that takes the data and generates 
#states from it

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





