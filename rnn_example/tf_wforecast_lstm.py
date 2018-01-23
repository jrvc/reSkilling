# -*- coding: utf-8 -*-
"""
Implementation of a RNN for weather forecasting.
Data downloaded from https://www.ncdc.noaa.gov/cdo-web/datatools/lcd for the tom green county

@author: Raul Vazquez
"""

#import dateutil.parser
#import datetime
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn

#from pymongo import MongoClient
#from bson.objectid import ObjectId

#from sklearn.metrics import mean_squared_error

import os
os.chdir("C:\\Users\\Raul Vazquez\\Desktop\\reSkilling\\reSkilling\\rnn_example")
from tf_lstm import lstm_model


''' 
-----------------------------------------------------
--------- pass these to an utils file: --------------
----------------------------------------------------- '''

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observations, suitable for RNNs.
    recall that RNNs observe x_1,...,x_n and try to guess x_{n+1}
    INPUT:
        - data: (pd.DataFrame) 
        - time_steps: (int) number of previous steps to take into account for every time step
        - labels: (boolean) True for y variable (objective var) and False for X variable (observed var)
    OUTPUT:
        - (np.array) 
            if labels == True: inputed data without the first time_steps observations
            if labels == False: array of size 'len(data) - time_steps'. Each entry contains an array
                                with the observation and its previous time_steps observations
    * example:
        l = pd.DataFrame([1, 2, 3, 4, 5])
        rnn_data(l , 2) = [[1, 2], [2, 3], [3, 4]]
        rnn_data(l , 2, True) =  [3, 4, 5]    
        rnn_data(l , 3) = [[1, 2, 3], [2, 3, 4]]
        rnn_data(l , 3, True) =  [4, 5]  
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

"""
def split_data(data, val_size=0.1, test_size=0.1):
    """
    #splits data to training, validation and testing sets
    """
    
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test
"""

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    INPUT:
        - data: (pd.DataFrame)
        - time_steps: (int)
        - labels: (boolean) True for y variable and False for X variable
        - val_size: (double) size of the validation set (proportion, between 0 and 1)
        - test_size: (double) size of the test set
    OUTPUT:
        - train, test and validation sets of sizes:
            train = data from timestep 0 to timestep len(data) - ( len(validation) + len(test) )
            val = data from len(train) to len(data) - len(test)
            test = last observations of the time series
          The sets are returnes in a suitable structure for a RNN (*refer to rnn_data)
    """
    # compute the size of the test and validation sets
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
    
    
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
    
    #df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))
    

def load_csvdata(rawdata, time_steps, seperate=False):
    '''
    Function to load the given CSV data
    INPUT:
         - rawdata: (DataFrame) the original data 
         - time_steps: (int) number of steps to consider 
         - seperate: (boolean) indicates if data is already divided into 
                     'a' = the X variables or observed variables; and 
                     'b' = the y variable or objective variable 
    OUTPUT:
        - X: dictionary containing the train, test and validation sets with observed variable
        - y: dictionary  "             "                "                "  objective variable
    '''
    data = rawdata
    # make sure we have a pandas Data Frame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # separate into train, test and validation sets
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

''' 
-----------------------------------------------------
-----------------------------------------------------
----------------------------------------------------- '''
from dateutil import parser

def load_weather_frame(filename):
    #loads only the dates and weather data
    data_raw = pd.read_csv(filename, dtype={'HOURLYWETBULBTEMPC': float, 'Date': str}, na_values={'HOURLYWETBULBTEMPC': []})
    df =  pd.DataFrame(data_raw, columns=['DATE','HOURLYWETBULBTEMPC'])
    df['DATE'] = [parser.parse(df['DATE'][i]) for i in range(len(df))]
    #df['HOURLYWETBULBTEMPC'].replace('NaN', np.NaN)
    return df.set_index('DATE')


TIMESTEPS = 10
data_weather = load_weather_frame("weather-db_tom-green-county.csv") 
# get rid of the NaN's
data_weather = data_weather[(~(data_weather['HOURLYWETBULBTEMPC'].isnull()))]
X, y = load_csvdata(data_weather, TIMESTEPS, seperate=False)

# specs
LOG_DIR = './ops_logs/lstm_weather'
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 1000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100



regressor = learn.SKCompat(learn.Estimator(
        model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
        model_dir=LOG_DIR
))
    
# create an lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predicted = regressor.predict(X['test'])
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))





# plot the data
all_dates = data_weather.index.get_values()

fig, ax = plt.subplots(1)
fig.autofmt_xdate()

predicted_values = predicted.flatten() #already subset
predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
plot_predicted, = ax.plot(pd.Series(predicted_values, index=predicted_dates), label='predicted (c)')

test_values = y['test'].flatten()
test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
plot_test, = ax.plot(pd.Series(test_values, index=test_dates), label='observed (c)')

xfmt = mdates.DateFormatter('%b %d, %Y')# %H:%M')
ax.xaxis.set_major_formatter(xfmt)

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('Tom Green county Weather Predictions')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()



''' LOOKS LIKE OVERFITTING!!!! '''
plt.plot(np.append(y['train'],y['test'])),plt.plot(np.append(y['train'],predicted))

plt.plot(regressor.predict(X['val'])),plt.plot(y['val'])