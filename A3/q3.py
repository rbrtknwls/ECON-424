import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
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
neigh = KNeighborsClassifier(n_neighbors=19,n_jobs=-1)

train, test = train_test_split(data, test_size=0.1)

Y_Train = train.iloc[:, 0]
X_Train = train.iloc[:, 1:]

Y_Test = test.iloc[:, 0]
X_Test = test.iloc[:, 1:]

neigh.fit(X_Train, Y_Train)

y_scores = neigh.predict_proba(X_Test)[:,1]
# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(Y_Test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc='lower right')
#plt.show()

# Define a range of thresholds to test
thresholds = np.linspace(0.0, 0.5, 100)

# Initialize lists to store error rates
total_error = []
false_negative = []
false_positive = []

# Calculate error rates for different thresholds
for threshold in thresholds:
    y_pred = (neigh.predict_proba(X_Test)[:, 1] >= threshold).astype(int)
    error_rate = 1 - accuracy_score(Y_Test, y_pred)
    total_error.append(error_rate)

    tn, fp, fn, tp = confusion_matrix(Y_Test, y_pred).ravel()
    
    false_negative_rate = fn / (fp + fn)
    false_postive_rate = fp / (fp + tp)

    false_negative.append(false_negative_rate)
    false_positive.append(false_postive_rate)

# Plot the relationship between threshold and error rate
plt.plot(thresholds, total_error)
plt.plot(thresholds, false_negative)
plt.plot(thresholds, false_positive)
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Threshold for k-NN Classifier')
plt.grid(True)
plt.show()