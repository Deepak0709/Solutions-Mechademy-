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

def neural_model():
    model = Sequential()
    model.add(Dense(48, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer= 'adam', loss = 'mse')

    return model