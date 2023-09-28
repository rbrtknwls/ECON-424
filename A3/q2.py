import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Read in the data
data = pd.read_csv("Econ_424_F2023_PC3_training_small.csv", low_memory=False)

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

trainingResults = []
testResults = []
valsOverK = []
for k in range(1, 100, 1):
    neigh = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)

    train, test = train_test_split(data, test_size=0.1)

    Y_Train = train.iloc[:, 0]
    X_Train = train.iloc[:, 1:]

    Y_Test = test.iloc[:, 0]
    X_Test = test.iloc[:, 1:]

    neigh.fit(X_Train, Y_Train)

    testPass = 0
    testError = 0

    trainingPass = 0
    trainingError = 0

    # Run tests of training data
    trainingPredict = neigh.predict(X_Train)
    for idx in range(0, len(trainingPredict)):
        if trainingPredict[idx] != Y_Train.iloc[idx]:
            trainingError += 1
        else:
            trainingPass += 1

    # Run tests of test data
    testPredict = neigh.predict(X_Test)
    for idx in range(0, len(testPredict)):
        if testPredict[idx] != Y_Test.iloc[idx]:
            testError += 1
        else:
            testPass += 1

    # Push the results onto an array to be printed later
    trainingResults.append(trainingError/(trainingPass+trainingError))
    testResults.append(testError/(testPass+testError))
    valsOverK.append(1 / k)

    print("====" + str(k) + "====")
    print("Training Results: " + str(trainingError/(trainingPass+trainingError)))
    print("Test Results: " + str(testError/(testPass+testError)))

plt.plot(valsOverK, trainingResults, label="Training Error")
plt.plot(valsOverK, testResults, label="Test Error")
plt.xlabel("1/K")
plt.ylabel("Error Rate")
plt.title("Effect of K on Test and Training Error Rate")
plt.legend()
plt.show()