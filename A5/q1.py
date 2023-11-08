import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")

# Read in the data
data = pd.read_csv("processed.csv", low_memory=False)

# Standardize results
data["price"] = data["price"].map(lambda x: np.log(x))

data = data[data["price"] > np.log(40000)]

catagories = ["body_type", "engine_type", "fleet", "frame_damaged", "franchise_dealer", "franchise_make",
              "fuel_type", "has_accidents", "iscab", "is_certified", "is_cpo", "is_new", "is_oemcpo",
              "make_name", "model_name", "salvage", "sp_name", "theft_title", "transmission_display",
              "trimid", "trim_name", "vehicle_damage_category", "wheel_system", "wheel_system_display"]


full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), catagories)])
innerModel = xgb.XGBRegressor(n_estimators=500, max_depth=6, eta=0.1, reg_lambda=0.1, colsample_bytree=0.4)

Y_Train = data.iloc[:, 0]
X_Train = data.iloc[:, 1:]
encoder = full_pipeline.fit(X_Train)
X_train = encoder.transform(X_Train)
innerModel.fit(X_train, Y_Train)


test = pd.read_csv("Econ424_F2023_PC5_test_data_without_response_var.csv", low_memory=False)

X_test = encoder.transform(test)

val = innerModel.predict(X_test)

f = open('predictions.csv', 'w')
for estimate in val:
    f.writelines(str(estimate) + ",\n")
'''


X_Train = [0]*10
Y_Train = [0]*10
X_Test = [0]*10
Y_Test = [0]*10

NUMTESTS = 10
for i in range(0,NUMTESTS):

    train, test = train_test_split(data, test_size=0.1)

    Y_Train[i] = train.iloc[:, 0]
    X_Train[i] = train.iloc[:, 1:]

    Y_Test[i] = test.iloc[:, 0]
    X_Test[i] = test.iloc[:, 1:]

innerModel = xgb.XGBRegressor(n_estimators=500, max_depth=6, eta=0.1, reg_lambda=0.1, colsample_bytree=0.4)
MSE = 0
DIF = 0
for i in range(0,NUMTESTS):
    encoder = full_pipeline.fit(X_Train[i])

    X_train = encoder.transform(X_Train[i])
    innerModel.fit(X_train, Y_Train[i])

    X_test = encoder.transform(X_Test[i])
    predicted = innerModel.predict(X_test)

    localMSE = 0
    localDIF = 0
    for x in range(0, len(predicted)-1):
        localMSE += (predicted[x] - Y_Test[i].iloc[x])**2
        localDIF += (Y_Test[i].iloc[x] - np.mean(Y_Test[i]))**2

    localMSE /= (len(predicted)-1)
    localDIF /= (len(predicted)-1)
    MSE += localMSE
    DIF += localDIF

MSE /= NUMTESTS
DIF /= NUMTESTS

print(MSE)
print(1 - MSE/DIF)
'''
