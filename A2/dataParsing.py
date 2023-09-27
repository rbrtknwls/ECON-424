import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

data = pd.read_csv("parsed.csv", low_memory=False)

data = data[(np.abs(stats.zscore(data)) < 4).all(axis=1)]

data.to_csv("parsing.csv", index=False)

#x = open("parsing.csv", "r")
#y = open("parsed.csv", "w")
#lines = x.readlines()

#for line in lines:
#    y.writelines(line.replace("\'", ""))

#data.drop(data.columns.difference(["VALUE","BATHS","BEDRMS","BUILT","UNITSF","LOT","ROOMS","REGION","KITCHEN","FLOORS","LAUNDY","RECRM","METRO","METRO3"]), axis=1, inplace=True)