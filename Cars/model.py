
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from model import model
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

def model():
	model = Sequential()
	model.add(Dense(48, activation= 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(24, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(24, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(12, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(12, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(optimizer = 'adam', loss = 'mse')
	
	return model