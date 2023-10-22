import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

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

bucketAverages = []
diffs = []
minix = []
factor = 100

for x in range(0, int(10000/factor)):
    minix.append(x*factor)

for bucket in range(0, 1601):

    if (bucket == 0):
        slice = data[
            (data['mileage'] > bucket*factor) & (data['mileage'] <= bucket*factor + factor)]
        test_y = slice.iloc[:, 0]
        test_x = slice.iloc[:, 1:-1]

        bucketAverages.append(test_y.mean())
        diffs.append(0)
    else:
        slice = data[
            (data['mileage'] > bucket*factor) & (data['mileage'] <= bucket*factor + factor)]


        test_y = slice.iloc[:, 0]
        test_x = slice.iloc[:, 1:-1]

        diffs.append(bucketAverages[-1] - test_y.mean())
        bucketAverages.append(test_y.mean())

# ===================== Nice Graph Making =====================
x_vals = []
minisum = []
mini_x = []
for bucket in range(0, 1601):
    x_vals.append((bucket*factor + factor) % 10000)


for x in range(0, int(10000/factor)):
    mini_x.append(x*factor)
    minisum.append(0)

for bucket in range(0, 1601):
    minisum[int(x_vals[bucket]/factor)] += diffs[bucket]

for x in range(0, int(10000/factor)):
    minisum[x] = minisum[x] / int((1601 * factor) / 10000)

print(x_vals)
print(bucketAverages)
#plt.plot(x_vals, diffs, '.b')
plt.plot(mini_x, minisum, '.r')
plt.xlabel("Amount of Miles Driven (mod 10000)")
plt.ylabel("Difference in Price of a Car (log)")
plt.title("Difference in Price of Car vs Miles Driven")
plt.show()
