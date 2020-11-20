from model_function import *  # model_funtion.py is the main program that calls model.py program.
import pandas as pd
import numpy as np

#input var.
work_dir = 'C:/Users/DK/OneDrive/Desktop/Coursera/Projects/Car_Predicitons_Final'     # current working directory
test_data_percentage = 10                                                             # to split data into train and test
do_data_visualisation = False
epochs = 5
batch_size = 128

#===================================================================================================================
# Loading Data
df_1 = pd.read_csv(work_dir + '/cars_price.csv', parse_dates = ['year'])
df_2 = data_mapping(df_1)   #removes redundant data and convert strings variables to numberical


if do_data_visualisation:
    
    # create_plot(X, plot-type, hue = False | True, data) for x-axis these are the options year, mileage(kilometers) or volume(cm3) and categorical columns. 
    # Plot type can be scatter(For priceUSD vs numerical) or bar(For priceUSD vs categorical columns).
    do_hue = False
    create_plot('year', 'scatter', do_hue, '_',df_2) # scatter plot - price vs year (without hue)
    
    do_hue = True
    create_plot('year', 'scatter', do_hue , 'drive_unit', df_2) # scatter plot - price vs year (with hue = drive_unit )
    
    do_hue = False
    create_plot('drive_unit', 'bar', do_hue, '_', df_2) # bar plot - price vs drive_unit (without hue)
    
    do_hue = True
    create_plot('drive_unit', 'bar', do_hue ,'fuel_type', df_2) # bar plot price vs drive_unit (with hue = 'fuel_type')

    
# splitting the data into train and test set.
X_train, X_test, y_train, y_test = split_train_test_data(df_2) 
    
#Scaling the data. 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

predictions, model = create_and_train_model(X_train, y_train, X_test, y_test, epochs, batch_size)
show_error(predictions, y_test)

#====================================================================================================================
#do predictions   (AN EXAMPLE SHOWN)

year = 2006                        # numerical variable
condition = 2                      #(1-3)
drive_unit = 1                     #(1-4)
mileage_kilometers = 173200.0      # float variable
fuel_type =2                       #(1-3)
volume_cm3= 1500                   # numerical variable
transmission = 2                   #(1-2)
segment = 4                        #(1 to 9)

data = [[year, condition, mileage_kilometers, fuel_type, volume_cm3, transmission, drive_unit, segment]]
data = np.array(data)
predict(data, model, scaler)       # calling predict function from model.py program

