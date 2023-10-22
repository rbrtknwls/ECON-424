import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from pandas.util import hash_pandas_object

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Read in the data
data = pd.read_csv("inter.csv", low_memory=False)

# Keep only the most relevant stats
data = data.loc[:,
       data.columns.intersection(['price', 'model_name', 'mileage', 'daysonmarket', 'owner_count', 'year'])]
# If owner_count is empty assume its 0
data["owner_count"] = data["owner_count"].fillna(0)
# Drop empty columns
data.dropna(inplace=True)
# Standardize results
data["price"] = data["price"].map(lambda x: np.log(x))
data["mileage"] = data["mileage"].map(lambda x: np.log(int(x) + 1))

# ===================== Data Processing =====================
# assign each value to a dict
modelDict = {}


def classify(model):
    if model not in modelDict:
        modelDict[model] = len(modelDict)


data.apply(lambda x: classify(x['model_name']), axis=1)
data["model_name"] = data["model_name"].map(lambda x: modelDict[x])

# We will now define 3 buckets dependent on the price of the car, such that if we estimate a car to be in a specific
#  range we will use the bucket estimate instead of the overall estimator.
data.sort_values(by=['price'], inplace=True, ignore_index=True)

numberOfBuckets = 5
numberOfEntries = len(data)
markers = [data.iloc[0, 0]]
adjustment = np.abs((data.iloc[0, 0] - data.iloc[numberOfEntries - 1, 0]) / 14)
for i in range(1, numberOfBuckets + 1):
    if i == numberOfBuckets:
        markers.append(data.iloc[numberOfEntries - 1, 0])
    else:
        markers.append(data.iloc[int(i * numberOfEntries / numberOfBuckets), 0])



# ===================== Data Testing =====================

def getModelName(modelName):
    if modelName in modelDict:
        return modelDict[modelName]
    return 0


# Start by predicting using a regular classifier, based of results use a more specific classifier
baseLine = xgb.XGBRegressor(n_estimators=500, max_depth=4, eta=0.1, subsample=0.7, colsample_bytree=0.8)
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
baseLine.fit(x, y)
prediction_base = baseLine.predict(x)


MSE = 0
# Using the old predictions, use the buckets to predict again
for bucket in range(0, numberOfBuckets):
    print(bucket)
    if bucket == numberOfBuckets - 1:
        slice = data[data['price'] > markers[bucket]]
    elif bucket == 0:
        slice = data[data['price'] <= markers[bucket + 1]]
    else:
        slice = data[
            (data['price'] > markers[bucket]) & (
                    data['price'] <= markers[bucket + 1])]

    slice.reset_index()
    loctest = slice.iloc[:, 1:]
    loctest.reset_index()
    predicted = baseLine.predict(loctest)

    localMSE = 0
    for idx in range(0, len(predicted)):
        localMSE += (slice.iloc[idx, 0] - predicted[idx])**2
    localMSE = localMSE/len(predicted)
    MSE += localMSE

MSE = MSE/numberOfBuckets
print(MSE)
print(MSE/y.var())
'''
bar_labels = ["1st Quartile", "2nd Quartile", "3rd Quartile", "4th Quartile", "5th Quartile"]
bar_colors = ['brown', 'blue', 'green', 'purple', 'orange']

plt.bar(["1st Quartile", "2nd Quartile", "3rd Quartile", "4th Quartile", "5th Quartile"], MSEList, label=bar_labels, color=bar_colors)
plt.ylabel('MSE for Predicted Price vs Actual Price (log)')
plt.title('MSE for Predicted Prices Across Price Quartile')
plt.legend(title='Price Quartile')
plt.show()
'''