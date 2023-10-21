import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
# Read in the data
data = pd.read_csv("Econ_424_F2023_PC3_training_large.csv", low_memory=False)

# =============== DATA PREPROCESSING ===============

# Modify some of the columns that we wont be using
data.drop(columns=["city", "state", "make"], inplace=True)

# Some models are the same but use weird capitalization or ., this fixes that
data["model"] = data["model"].apply(lambda x: (str(x).lower()).split(".")[0])

# This function will keep a count of each model of car if its below or above 18400
#    ModelDict[model] = (#num cars above 18400, #num cars below 18400)
ModelDict = {}


def classify(price, dep):
    # Cluster cars without considering values after the .
    dep = dep.split(".")[0]

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

# This is the factor we use for assigning a numeric value to each car model, since from
#   looking at the data the model has a much greater fit with the price then the milage the
#   difference in model is equal to a value greater than the difference in mileage
diffFactor = 10 ** (int(np.log10(data["mileage"].max())) + 1)

# Since the number of large clusters is dynamic we start at 3 and keep growing
currLargeidx = 3
ModelWeights = {}

for key in ModelDict:
    if ModelDict[key][0] + ModelDict[key][1] < 30:
        # Only < 30 observations, cluster with similar small sample models

        if ModelDict[key][0] / (ModelDict[key][0] + ModelDict[key][1]) > 0.9:
            # Key belongs to above cluster (small)
            ModelWeights[key] = 2 * diffFactor
        elif ModelDict[key][1] / (ModelDict[key][0] + ModelDict[key][1]) > 0.9:
            # Key belongs to below cluster (small)
            ModelWeights[key] = 1 * diffFactor
        else:
            # Key belongs to either cluster (small)
            ModelWeights[key] = 0
    else:
        # Key belongs to its own cluster (enough observations)
        ModelWeights[key] = currLargeidx * diffFactor
        currLargeidx += 1

# Change the value of each model using the created dictionary
data["model"] = data["model"].map(lambda x: ModelWeights[x])
# Change the price to be 1 or 0 depending on price
data["price"] = data["price"].apply(lambda x: 1 if x < 18400 else 0)
# Change the year so that it the marginal year difference is worth more
data["year"] = data["year"].map(lambda x: x * (diffFactor / 10))

# =============== DATA TRAINING ===============


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

absError = 0
absCorrect = 0

predicted1Actual1 = 0
predicted1Actual0 = 0
predicted0Actual1 = 0
predicted0Actual0 = 0
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



# =============== RUNNING THE PREDICTION ===============

Y_Train = data.iloc[:, 0]
X_Train = data.iloc[:, 1:]

neigh.fit(X_Train, Y_Train)

X_Test = pd.read_csv("Econ_424_F2023_PC3_test_without_response_variable.csv", low_memory=False)
X_Test.drop(columns=["city", "state", "make"], inplace=True)

def castModel(value):
    if value == "mkxpremiere":
        return ModelWeights["mkxmkx"]
    if "s70" in value:
        return ModelWeights["s704dr"]
    if "lacrosse" in value:
        return ModelWeights["lacrossefwd"]
    if "routansel" in value:
        return ModelWeights["routan4dr"]
    if "caliber" in value:
        return 0
    if "eclipse" in value:
        return ModelWeights["eclipse2dr"]
    if "equinox" in value:
        return ModelWeights["equinoxawd"]
    if value in ModelWeights:
        return ModelWeights[value]
    else:
        # Assume it's in the pool of small sample observations
        return 0


X_Test["model"] = X_Test["model"].apply(lambda x: (str(x).lower()).split(".")[0])
X_Test["model"] = X_Test["model"].map(lambda x: castModel(x))
X_Test["year"] = X_Test["year"].map(lambda x: x * (diffFactor / 10))

estimates = neigh.predict(X_Test)

# Write the output
f = open('predictions.csv', 'w')
for estimate in estimates:
    f.writelines(str(estimate) + ",\n")
