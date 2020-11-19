from propulsion_function import *
import pandas as pd

# Input variables
work_dir = 'C:/Users/DK/OneDrive/Desktop/Coursera/Projects/Propulsion'  # present working directory
df = pd.read_csv(work_dir + '/propulsion.csv')                          # loading data 
epoch = 30
batch = 128

#===========================================================================================================================

# NOTE: There are two columns to predict namely GT Compressor decay state coefficient, GT Turbine decay state coefficient,
#       in the sample example beow the models are predicting GT Compressor decay state coefficient, for predicting the 
#       just un-comment the second exmample which is commneted.


X,y=data(df, 'GT Compressor decay state coefficient.') #The second argument is the column we want to predict out of the two,
                                                   #GT Compressor decay state coefficient, GT Turbine decay state coefficient.

feature_selector(X, y)
X = imp_data(X)
X_train, X_test, y_train, y_test = data_split(X, y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#the last argument in the functin below is the type of model we wish to use among Linear, Ridge, Lasso
predictions,error,coefficients,intercept, mae, rmse, model = create_model(X_train, X_test, y_train, y_test, 'Linear')  
                                                                                             
#We can use a neural network also for predicting the target.
#predictions_n,error_n,weight_n, mae, rmse, model_n = neural_network(X_train, X_test, y_train, y_test, epoch, batch) 

#============================================= (AN EXAMPLE PREDICTION)===================================================

# Data to predict GT Compressor decay state coefficient, for predicitng GT Turbine decay state coefficient provide the value
# of GT Compressor decay state coefficient in place of GT Turbine decay state coefficient in the data below.

Lever_position_lp  = 3.144
Ship_speed_v_knots = 18
Gas_Turbine_shaft_torque_GTT_kNm = 1386.739
Gas_Turbine_rate_of_revolutions_GTn_rpm = 7051.012
Gas_Generator_rate_of_revolutions_GGn_rpm = 8278.024
Starboard_Propeller_Torque_Ts_kN = 60.339
Port_Propeller_Torque_Tp_kN = 60.806
HP_Turbine_exit_temperature_T48_C = 769.651
GT_Compressor_outlet_air_temperature_T2_C = 576.304
HP_Turbine_exit_pressure_P48_bar = 1.518
GT_Compressor_outlet_air_pressureP2_bar = 7.374
Gas_Turbine_exhaust_gas_pressure_Pexh_bar = 1.031
Turbine_Injecton_Control_TIC_percentage = 11.514
Fuel_flow_mf_kg_per_second = 0.676
GT_Turbine_decay_state_coefficient = 0.978

data = [[Lever_position_lp, Ship_speed_v_knots, Gas_Turbine_shaft_torque_GTT_kNm, 
         Gas_Turbine_rate_of_revolutions_GTn_rpm, Gas_Generator_rate_of_revolutions_GGn_rpm,
         Starboard_Propeller_Torque_Ts_kN, Port_Propeller_Torque_Tp_kN, HP_Turbine_exit_temperature_T48_C,
         GT_Compressor_outlet_air_temperature_T2_C, HP_Turbine_exit_pressure_P48_bar, 
         GT_Compressor_outlet_air_pressureP2_bar, Gas_Turbine_exhaust_gas_pressure_Pexh_bar, 
         Turbine_Injecton_Control_TIC_percentage, Fuel_flow_mf_kg_per_second, GT_Turbine_decay_state_coefficient]]

data = np.array(data)
predict(data, model_n, scaler) # In second argument provide the model name which was used to made predicitons on test data.
#=======================================================================================================================





#=========================FOR PREDICTION OF "GT Turbine decay state coefficient" on test set============================
#                                       (Just un-comment the code)

#The second argument is the column we want to predict out of the two,
#GT Compressor decay state coefficient and GT Turbine decay state coefficient

#X,y=data(df, 'GT Turbine decay state coefficient.') 
                                                   
#feature_selector(X, y)
#X = imp_data(X)
#X_train, X_test, y_train, y_test = data_split(X, y)

#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#the last argument in the functin below is the type of model we wish to use among Linear, Ridge, Lasso
#predictions,error,coefficients,intercept, model = create_model(X_train, X_test, y_train, y_test, 'Lasso')  
                                                                                             
#We can use a neural network also for predicting the target.
#predictions_n,error_n,weight_n, model_n = neural_network(X_train, X_test, y_train, y_test, epoch, batch) 


#=========================(FOR PREDICTING GT Turbine decay state coefficient. on user defined data)===========================
#                                       (Just un-comment the code)

# Data to predict GT Compressor decay state coefficient, for predicitng GT Turbine decay state coefficient provide the value
# of GT Compressor decay state coefficient in place of GT Turbine decay state coefficient in the data below.

# Provide the reasonable values for each feature.

# Lever_position_lp  = 
# Ship_speed_v_knots = 
# Gas_Turbine_shaft_torque_GTT_kNm = 
# Gas_Turbine_rate_of_revolutions_GTn_rpm = 
# Gas_Generator_rate_of_revolutions_GGn_rpm = 
# Starboard_Propeller_Torque_Ts_kN = 
# Port_Propeller_Torque_Tp_kN = 
# HP_Turbine_exit_temperature_T48_C = 
# GT_Compressor_outlet_air_temperature_T2_C = 
# HP_Turbine_exit_pressure_P48_bar = 
# GT_Compressor_outlet_air_pressureP2_bar = 
# Gas_Turbine_exhaust_gas_pressure_Pexh_bar = 
# Turbine_Injecton_Control_TIC_percentage = 
# Fuel_flow_mf_kg_per_second = 
# GT_Compressor_decay_state_coefficient = 

# data = [[Lever_position_lp, Ship_speed_v_knots, Gas_Turbine_shaft_torque_GTT_kNm, 
#          Gas_Turbine_rate_of_revolutions_GTn_rpm, Gas_Generator_rate_of_revolutions_GGn_rpm,
#          Starboard_Propeller_Torque_Ts_kN, Port_Propeller_Torque_Tp_kN, HP_Turbine_exit_temperature_T48_C,
#          GT_Compressor_outlet_air_temperature_T2_C, HP_Turbine_exit_pressure_P48_bar, 
#          GT_Compressor_outlet_air_pressureP2_bar, Gas_Turbine_exhaust_gas_pressure_Pexh_bar, 
#          Turbine_Injecton_Control_TIC_percentage, Fuel_flow_mf_kg_per_second, GT_Compressor_decay_state_coefficient]]
# data = np.array(data)
# predict(data, model, scaler) # provide the model which was used to made predicitons on test data.





