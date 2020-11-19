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

def model(model_type):
    if model_type == 'Linear':                             # a linear model, a ridge model and a lasso model.
        print('Training a Linear model...')
        model = LinearRegression()
        print('Model trained!')
              
    if model_type == 'Ridge': 
        print('Training a Ridge model...')
        model = Ridge(alpha=0.0001)
        print('Model trained!')
        
    if model_type == 'Lasso':
        print('Training a Lasso model...')
        model = Lasso(alpha = 0.0001)
        print('Model trained!')
    return model