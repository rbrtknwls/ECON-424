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

regr_1 = DecisionTreeRegressor(max_depth=6, min_samples_leaf=50,min_samples_split=19)

Y_Train = data.iloc[:, 0]
X_Train = data.iloc[:, 1:]

X_Test = pd.read_csv("Econ424_F2023_PC2_test_set_without_response_variable_v1.csv", low_memory=False)
X_Test.rename(str.upper, axis='columns', inplace=True)

regr_1.fit(X_Train, Y_Train)


plt.figure(figsize=(10, 5))

plt.bar(regr_1.feature_names_in_, regr_1.feature_importances_, color='black',
        width=0.4)

plt.xlabel("Features in regression tree")
plt.ylabel("Gini importance")
plt.title("Relative Feature Importance")
plt.show()