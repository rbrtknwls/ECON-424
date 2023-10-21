import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Read in the data
'''
data = pd.read_csv("Econ424_F2023_PC4_training_data_large.csv", low_memory=False)
data = data.loc[:, data.columns.intersection(['price', 'make_name', 'mileage', 'model_name', 'year'])]
data.dropna(inplace=True)
'''
data = pd.read_csv('inter.csv')

# Keep only the most relevant
data = data.loc[:, data.columns.intersection(['price', 'mileage', 'model_name', 'year'])]
data["price"] = data["price"].map(lambda x: np.log(x))
data["mileage"] = data["mileage"].map(lambda x: np.log(int(x) + 1))
data, test = train_test_split(data, test_size=0.998)

# ===================== Data Processing =====================
# Get a count of the number of models
modelDictCount = {}


def classify(model):
    if model not in modelDictCount:
        modelDictCount[model] = 0

    modelDictCount[model] = 1 + modelDictCount[model]


data.apply(lambda x: classify(x['model_name']), axis=1)

# Now we want to classify the models into groups
modelToGroup = {}
numberOfGroups = 1

for x in modelDictCount:
    # We want at least 20 observations per variate, so we will make sure each group has 40 samples
    if modelDictCount[x] > 40:
        modelToGroup[x] = numberOfGroups
        numberOfGroups += 1

    else:
        modelToGroup[x] = 0

data["model_name"] = data["model_name"].map(lambda x: modelToGroup[x])
# We are now going to train XGBoost on each group

modelPerGroup = {}

print(numberOfGroups)
for idx in range(0, numberOfGroups + 1):
    modelDataFrame = data[data['model_name'] == idx]
    x = modelDataFrame.iloc[:, 1:]
    y = modelDataFrame.iloc[:, 0]
    model = xgb.XGBRegressor(n_estimators=500, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model.fit(x, y)
    modelPerGroup[idx] = model
    if idx % 100 == 0:
        print(idx)


def getModelName(modelName):
    if modelName in modelToGroup:
        return modelToGroup[modelName]
    return 0


# ===================== Data Testing =====================

def getMSE(row):
    test_y = row[0]
    test_x = row[1:]
    test_x['model_name'] = getModelName(test_x['model_name'])
    pred_y = modelPerGroup[test_x['model_name']].predict(np.array([test_x]))

    print(pred_y, test_y)


test.apply(getMSE, axis=1)
'''
for index, row in test.iterrows():
    test_y = row[0]
    test_x = row[1:]
    test_x['model_name'] = getModelName(test_x['model_name'])
    print(test_x)
    print(test_x.model_name.dtypes)
    pred_y = modelPerGroup[test_x['model_name']].predict(test_x)

    print(pred_y, test_y)
'''
'''

dataFramePerModel = {}
for model in range(0, len(makeDict)):
    modelDataFrame = data[data['model_name'] == modelDict[list(modelDict.keys())[model]]]
    print(len(modelDataFrame))
    dataFramePerModel[model] = modelDataFrame

'''
'''
model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

model.fit(trainingParams, trainingResponse)
estimate = model.predict(trainingParams)

MSE = 0
for x in range(0, len(estimate)):
    MSE += (estimate[x] - trainingResponse.iloc[x])**2

print(MSE)
print(len(estimate))

plt.plot(trainingParams["mileage"], estimate, 'b.', label="Training Error")
plt.plot(trainingParams["mileage"], trainingResponse, 'r.', label="Training Error")
plt.show()
'''

'''
X_Train = [0] * 10
Y_Train = [0] * 10
X_Test = [0] * 10
Y_Test = [0] * 10

for i in range(0, 10):
    train, test = train_test_split(data, test_size=0.1)

    Y_Train[i] = train.iloc[:, 0]
    X_Train[i] = train.iloc[:, 1:]

    Y_Test[i] = test.iloc[:, 0]
    X_Test[i] = test.iloc[:, 1:]

neigh = KNeighborsClassifier(n_neighbors=19)

MSE = []
for i in range(0, 10):
    neigh.fit(X_Train[i], Y_Train[i])
    predict = neigh.predict(X_Test[i])

    for idx in range(0, len(predict)):

        if predict[idx] == 0:
            if (Y_Test[i].iloc[idx] == 0):
                predicted0Actual0 += 1
            else:
                predicted0Actual1 += 1
        else:
            if (Y_Test[i].iloc[idx] == 0):
                predicted1Actual0 += 1
            else:
                predicted1Actual1 += 1

        # Calculations for Total Error
        if predict[idx] != Y_Test[i].iloc[idx]:
            absError += 1
        else:
            absCorrect += 1

print("Predicted:      0          1")
print("Actual [0]    " + str(predicted0Actual0) + "    " + str(
   predicted1Actual0))
print("Actual [1]    " + str(predicted0Actual1) + "   " + str(
    predicted1Actual1))
'''
