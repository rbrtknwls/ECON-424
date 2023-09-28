import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

data = pd.read_csv("Econ424_F2023_PC2_training_set_v1.csv", low_memory=False)

# Clean Data
data = data[(np.abs(stats.zscore(data)) < 4).all(axis=1)]

#  =============== PART 1 (Test the model) ===============
X_Train = [0]*100
Y_Train = [0]*100
X_Test = [0]*100
Y_Test = [0]*100

# Load up an array of test and training data

for i in range(0,100):
    train, test = train_test_split(data, test_size=0.1)

    Y_Train[i] = train.iloc[:, 0]
    X_Train[i] = train.iloc[:, 1:]
    Y_Test[i] = test.iloc[:, 0]
    X_Test[i] = test.iloc[:, 1:]

mse = [0]*100
TSS = [0]*100
RSS = [0]*100
regr_1 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50,min_samples_split=19)
for i in range(0,100):
    regr_1.fit(X_Train[i], Y_Train[i]) # Fit the regression tree with the corresponding values

    predicted_vals = regr_1.predict(X_Test[i])
    r2_mean = np.mean(Y_Test[i].iloc[i])
    for x in range(0, len(predicted_vals)):
        # Calculation for MSE
        mse[i] += ((predicted_vals[x] - Y_Test[i].iloc[x])**2)/len(predicted_vals)
        # Calculation for RSS
        RSS[i] += ((predicted_vals[x] - Y_Test[i].iloc[x])**2)
        TSS[i] += ((r2_mean - Y_Test[i].iloc[x])**2)

print("MSE: " + str(np.mean(mse)))
print("R2: " + str(1-(np.mean(RSS)/np.mean(TSS))))
#  =============== PART 2 (Predict the data) ===============

# Load Data for Prediction In
X_Test = pd.read_csv("Econ424_F2023_PC2_test_set_without_response_variable_v1.csv", low_memory=False)
X_Test.rename(str.upper, axis='columns', inplace=True)

Y_Train = data.iloc[:, 0]
X_Train = data.iloc[:, 1:]

# Train the decision tree with the optimal values

regr_1.fit(X_Train, Y_Train)

estimates = regr_1.predict(X_Test)

# Write the output
f = open('predictions.csv', 'w')
for estimate in estimates:
    f.writelines(str(estimate) + ",\n")
