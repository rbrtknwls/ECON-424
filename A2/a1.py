import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

'''
   Testing to determine the best parameters to use for our decision tree.
   The factors we want to optimize for are
       1) Max Depth
       2) MinLeafSize
       3) MinSampleSplit

results= []


X_Train = [0]*10
Y_Train = [0]*10
X_Test = [0]*10
Y_Test = [0]*10

for i in range(0,10):

    train, test = train_test_split(data, test_size=0.1)

    Y_Train[i] = train.iloc[:, 0]
    X_Train[i] = train.iloc[:, 1:]

    Y_Test[i] = test.iloc[:, 0]
    X_Test[i] = test.iloc[:, 1:]
regr_1 = DecisionTreeRegressor()
for maxDepth in range(5,7):
    for minLeafSize in range(40,70):
        for minSplitSize in range(3,20):
            regr_1.set_params(**{"max_depth": maxDepth, "min_samples_leaf": minLeafSize, "min_samples_split": minSplitSize})

            mse = 0
            for i in range(0,10):
                regr_1.fit(X_Train[i], Y_Train[i])
                vals = regr_1.predict(X_Test[i])
                for x in range(0, len(vals)):
                    mse += ((vals[x] - Y_Test[i].iloc[x])**2)

            # Calculate Error in Samples and report it back

            results.append((int(mse), str(maxDepth) + " | " + str(minLeafSize) + " | " + str(minSplitSize)))


results.sort(key=lambda a: a[0], reverse=True)

x = True
first = 0
while x:
    val = results.pop()
    if (first == 0):
        first = val[0]
    if (val[0] != first):
        x= False
    print(val)

for x in range(0,20):
    print(results.pop())
    
====== RESULTS ====
Best Depth: 6
Best Min Leaf Size: 50
Best Split: 19
'''

data = pd.read_csv("Econ424_F2023_PC2_training_set_v1.csv", low_memory=False)

# Clean Data
data[(np.abs(stats.zscore(data)) < 4).all(axis=1)]

# Load Data for Prediction In
X_Test = pd.read_csv("Econ424_F2023_PC2_test_set_without_response_variable_v1.csv", low_memory=False)
X_Test.rename(str.upper, axis='columns', inplace=True)
print(X_Test.columns)
Y_Train = data.iloc[:, 0]
X_Train = data.iloc[:, 1:]

# Train the decision tree with the optimal values
regr_1 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, min_samples_split=19)
regr_1.fit(X_Train, Y_Train)

estimates = regr_1.predict(X_Test)

# Write the output
f = open('predictions.csv', 'w')
for estimate in estimates:
    f.writelines(str(estimate) + ",\n")