import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam



def data(data, value):                             # value is the target column that we want to predict
    data = data.drop('Unnamed: 0', axis = 1)       # dropping redundant column.
    X = data.drop(value, axis = 1)                 # dropping te target column from data and saving to X
    y = data[value]                                # saving the target column to y.
    return X, y
    
                   
def feature_selector(X,y):                         # a function to tell the most related features with the target.
    feature_select = ExtraTreesRegressor()
    feature_select.fit(X, y)
    print('The correlation of features with target: \n')
    print(feature_select.feature_importances_)
    important_features = pd.Series(feature_select.feature_importances_, index = X.columns)
    important_features.nlargest(15).plot(kind = 'bar')
    print('\n')

def imp_data(X):
    X = X.drop(['GT Compressor inlet air temperature (T1) [C]', 'GT Compressor inlet air pressure (P1) [bar]'], axis = 1) # droping least related features.
    return X

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test
   

def create_model(X_train, X_test, y_train, y_test, model_type):   # function contains three different models for predictions
    from model import model
    model = model(model_type)     # argument has to be provided among Linear, Lasso, Ridge (Case Sensitive).
            
    model.fit(X_train, y_train) 
    predictions = model.predict(X_test)
    
    print('Making predictions and calculating errors...')
    error = y_test.values.reshape(3000, 1) - predictions
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    coefficients = model.coef_
    intercept = model.intercept_
    
    errors = pd.DataFrame({'Y': y_test, 'Predictions': predictions})
    plt.figure(figsize = (8,8))
    sns.set_style('darkgrid')
    sns.lmplot(x = 'Predictions', y = 'Y', data = errors)
    
    plt.figure(figsize = (8,8))
    sns.set_style('darkgrid')
    sns.distplot(error)
    print('\n')
    print('The mean absolute error of model is:' + str(mean_absolute_error(y_test, predictions)))
    print('The root mean squared error of model is:' + str(np.sqrt(mean_squared_error(y_test, predictions))))
    print('\n')
    return predictions, error, coefficients, intercept, mae, rmse, model
    
def neural_network(X_train, X_test, y_train, y_test, epoch, batch):    # a seprate neural network.
    from neural_model import neural_model
    model = neural_model()

    print('Training Neural Network...')
    model.fit(x = X_train, y = y_train.values,
              validation_data = (X_test, y_test.values),
              epochs = epoch, batch_size = batch)
              
    print('\n')          
    print('Neural Network trained!')
    loss = pd.DataFrame(model.history.history)
    sns.set_style('darkgrid')
    plt.figure(figsize = (8,8))
    loss.plot()

    print('Making predictions...')
    predictions= model.predict(X_test)

    error = y_test.values.reshape(3000, 1) - predictions
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    sns.set_style('darkgrid')
    plt.figure(figsize = (8,8))
    sns.distplot(error, kde = True)

    weights = model.weights
    print('\n')
    print('The mean absolute error of model is:' + str(mean_absolute_error(y_test, predictions)))
    print('The root mean squared error of model is:' + str(np.sqrt(mean_squared_error(y_test, predictions))))

    return predictions, error, weights, mae, rmse, model

def predict(test_sample, model, scaler):   
    #n = 10
    #test_sample =  test_sample.drop('priceUSD', axis = 1).iloc[n]
    #test_sample = scaler.transform(test_sample.values.reshape(-1,15))  # for rnd while using example from test sample
    test_sample = scaler.transform(test_sample.reshape(-1,15))
    output = model.predict(test_sample)[0][0]
    #print('The original price of car is:' +  str(data['priceUSD'].iloc[n]))
    print('\n')
    print('The expected decay coeeficent is: ' + str(output))


