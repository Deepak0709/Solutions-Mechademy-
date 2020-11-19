

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam



def create_plot(x, plot_type, do_hue, hue, data):
    print('Creating Plot...')
    #customizable input var.
    plt.figure(figsize = (8,6))
    sns.set_style('darkgrid')


    if plot_type == 'scatter' and do_hue == False :
            sns.scatterplot(x, y = 'priceUSD',data=data)

    if plot_type == 'scatter' and do_hue == True :
            sns.scatterplot(x, y = 'priceUSD',hue = hue, data = data, palette = 'viridis')
            
    if plot_type == 'bar' and do_hue == False :
            sns.barplot(x, y='priceUSD' , data = data)
            
    if plot_type == 'bar' and do_hue == True :
            sns.barplot(x, y='priceUSD', data= data, hue = hue, palette = 'viridis')
            

def show_header(data):
    print(data.head())
    
# Cleaning Data for plotting purpose.
def data_cleaning(data, do_show_header):
    print('Cleaning Data.')
    data['year'] = pd.to_datetime(data['year'])  
    data['year'] = data['year'].apply(lambda x: x.year)
    data = data.drop('Unnamed: 0', axis = 1)
    data = data.drop(['make', 'model', 'color'], axis = 1)
    data['volume(cm3)'].fillna(2000, inplace = True)
    data['drive_unit'].fillna(method = 'ffill', inplace = True)
    data['segment'].fillna(value = 'D', inplace = True)
    
    if do_show_header:
        print('Showing top 5 entries of cleaned data')
        show_header(data)
    return data 

def data_mapping(data):
    print('Cleaning and Mapping raw data...')
    do_show_header = False
    data = data_cleaning(data, do_show_header)
    
    #Converting categorical columns to numerical columns. 
    data['condition'] = data['condition'].map({'with mileage': 3, 'with damage': 2 , 'for parts':1})
    data['fuel_type'] = data['fuel_type'].map({'electrocar': 3, 'diesel':2, 'petrol': 1})
    data['transmission'] = data['transmission'].map({'mechanics': 1, 'auto': 2})
    data['segment'] = data['segment'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E':5, 'M':6, 'F':7, 'J': 8, 'S':9})
    data['drive_unit'] = data['drive_unit'].map({'front-wheel drive':1, 'rear drive': 2, 'part-time four-wheel drive':3,'all-wheel drive': 4})
    
    print('Showing top 5 entries of cleaned data')
    show_header(data)
    
    return data

# Splitting the data into train and test set.
def split_train_test_data(data):

    X = data.drop('priceUSD', axis=1)
    y = data['priceUSD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return  X_train, X_test, y_train, y_test


# Creating and training the model.
def create_and_train_model(train_X, train_y, test_X, test_y, epoch, batch):
    from model import model
    print('Training model ..')
    model = model()
# Training the model on 90% of the dataset (train data)
    model.fit(x= train_X, y = train_y.values,
          validation_data = (test_X, test_y.values), 
          epochs = epoch, batch_size = batch)
          
    loss = pd.DataFrame(model.history.history)
    loss.plot()

# Predicting the output (prieUSD) for test set.
    preds = model.predict(test_X)
    return preds, model
    print('Training completed.')


    
def show_error(predictions, y_test):  #gives error

    print('Calculating error.')
    print('The mean absolute ereor is: ' + str(mean_absolute_error(y_test, predictions)))
    print('The root mean squared error is: ' + str(np.sqrt(mean_squared_error(y_test, predictions))))
    
    
    
    
# Predicting priceUSD of a sample.
def predict(test_sample, model, scaler):   
    #n = 10
    #test_sample =  test_sample.drop('priceUSD', axis = 1).iloc[n]
    #test_sample = scaler.transform(test_sample.values.reshape(-1,8))  # for rnd while using example from test sample
    test_sample = scaler.transform(test_sample.reshape(-1,8))
    output = model.predict(test_sample)[0][0]
    #print('The original price of car is:' +  str(data['priceUSD'].iloc[n]))
    print('The expected price of car is: ' + str(output))




