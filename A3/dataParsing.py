import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

ModelDict = {}

data = pd.read_csv("Econ_424_F2023_PC3_training_large.csv", low_memory=False)

data.drop(columns=["city", "state", "make"], inplace=True)

def classify(price, dep):
    above = 0
    below = 0
    if price >= 18400:
        above = 1
    else:
        below = 1

    if not dep in ModelDict:
        ModelDict[dep] = (above, below)
    else:
        ModelDict[dep] = (ModelDict[dep][0] + above, ModelDict[dep][1] + below)


data.apply(lambda x: classify(x['price'], x['model']), axis=1)

diffFactor = 10**(int(np.log10(data["mileage"].max()))+1)

currLargeidx = 3
ModelWeights = {}

MaxError = 500000

for key in ModelDict:
    if ModelDict[key][0] + ModelDict[key][1] < 30:

        if ModelDict[key][0] / (ModelDict[key][0] + ModelDict[key][1]) > 0.7:
            # Key belongs to above cluster (small)
            ModelWeights[key] = 0
        elif ModelDict[key][1] / (ModelDict[key][0] + ModelDict[key][1]) > 0.7:
            # Key belongs to below cluster (small)
            ModelWeights[key] = 1 * diffFactor
        else:
            # Key belongs to either cluster (small)
            ModelWeights[key] = 2 * diffFactor
    else:
        # Key belongs to its own cluster (enough observations)
        ModelWeights[key] = currLargeidx * diffFactor
        currLargeidx += 1

data["model"] = data["model"].map(lambda x: ModelWeights[x])
data["price"] = data["price"].apply(lambda x: 1 if x < 18400 else 0)

for yearChange in range(10, 10000, 1000):

    newData = data
    newData["year"] = data["year"].map(lambda x: x * (diffFactor / yearChange))

    X_Train = [0]*10
    Y_Train = [0]*10
    X_Test = [0]*10
    Y_Test = [0]*10

    for i in range(0,10):

        train, test = train_test_split(newData, test_size=0.1)

        Y_Train[i] = train.iloc[:, 0]
        X_Train[i] = train.iloc[:, 1:]

        Y_Test[i] = test.iloc[:, 0]
        X_Test[i] = test.iloc[:, 1:]

    neigh = KNeighborsClassifier(n_neighbors=19)
    absError = 0
    absCorrect = 0
    for i in range(0, 10):
        neigh.fit(X_Train[i], Y_Train[i])
        predict = neigh.predict(X_Test[i])

        for idx in range(0, len(predict)):
            if predict[idx] != Y_Test[i].iloc[idx]:
                absError += 1
            else:
                absCorrect += 1

    if (absError < MaxError):
        MaxError = absError
        print("NEW TERM!")
        print(absError)
        print(absCorrect)
        print(round(absCorrect/(absCorrect+absError), 2))
        print("NEW BEST: " + str(yearChange))
    else:
        print("DivisFactor: " + str(yearChange) + " FAILED")