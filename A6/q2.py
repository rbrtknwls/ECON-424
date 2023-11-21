import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import re
import gensim
import seaborn as sn

from tensorflow.keras import layers, models
from gensim.models import KeyedVectors
from sklearn import metrics
dataset = pd.read_csv("Econ424_F2023_PC6_glassdoor_training_large_v1.csv")

firmDict = {}
jobTitleDict = {}
yearDict = {}
smallDict = {}
dateDict = {}

sentences = []


def convertString(string):
    return (''.join(ch for ch in string if ch.isalnum())).lower()


def updateDicts(row):
    firm = row["firm"]
    job = row["job_title"]
    year = row["year"]
    small = row["small"]
    date = row["date_review"]

    firm = convertString(firm)
    job = convertString(job)

    if firm not in firmDict:
        firmDict[firm] = 1
    else:
        firmDict[firm] = firmDict[firm] + 1

    if job not in jobTitleDict:
        jobTitleDict[job] = 1
    else:
        jobTitleDict[job] = jobTitleDict[job] + 1

    if year not in yearDict:
        yearDict[year] = 1
    else:
        yearDict[year] = yearDict[year] + 1

    if small not in smallDict:
        smallDict[small] = 1
    else:
        smallDict[small] = smallDict[small] + 1

    if year not in yearDict:
        yearDict[year] = 1
    else:
        yearDict[year] = yearDict[year] + 1

    if date not in dateDict:
        dateDict[date] = 1
    else:
        dateDict[date] = dateDict[date] + 1

def convertDayTime(date):
    if "/" in date:
        date = date.split("/")
    else:
        date = date.split("-")

    year = int(date[0])
    month = int(date[1])
    day = int(date[2])

    return day + month * 100 + year * 10000


def cleanAllStringsAndAdd(string):
    if type(string) == type("abc"):
        newString = re.sub('[^A-Za-z]+', ' ', string).lower()
        sentences.append(newString)
        return newString
    return ""


dataset.apply(updateDicts, axis=1)

dataset["firm"] = dataset['firm'].apply(lambda x: firmDict[convertString(x)] / len(firmDict))
dataset["job_title"] = dataset['job_title'].apply(lambda x: jobTitleDict[convertString(x)] / len(jobTitleDict))
dataset["date_review"] = dataset['date_review'].apply(lambda x: convertDayTime(x) / 2023119)
dataset["year"] = dataset['year'].apply(lambda x: x / 2023)

dataset["pros"] = dataset["pros"].apply(lambda x: cleanAllStringsAndAdd(x))
dataset["headline"] = dataset["headline"].apply(lambda x: cleanAllStringsAndAdd(x))
dataset["cons"] = dataset["cons"].apply(lambda x: cleanAllStringsAndAdd(x))

dataset.drop(columns=["location"], inplace=True)

dictOfEncodings = KeyedVectors.load_word2vec_format("google.bin", binary=True)


updatedDataFrame = []

def dataFrameToNpArray(row):
    prosVec = np.zeros(300)
    consVec = np.zeros(300)
    headLineVec = np.zeros(300)

    if row["pros"] is not None:
        rowWords = row["pros"].split(" ")
        for word in rowWords:
            try:
                word = dictOfEncodings[word]
                prosVec = np.add(word, prosVec)

            except KeyError:
                pass

    if row["cons"] is not None:
        rowWords = row["cons"].split(" ")
        for word in rowWords:
            try:
                word = dictOfEncodings[word]
                consVec = np.add(word, prosVec)

            except KeyError:
                pass

    if row["headline"] is not None:
        rowWords = row["headline"].split(" ")
        for word in rowWords:
            try:
                word = dictOfEncodings[word]
                headLineVec = np.add(word, prosVec)

            except KeyError:
                pass


    otherFeatures = np.array([row["firm"], row["job_title"], row["date_review"], row["year"], row["small"]])
    finalArray = np.asarray(np.concatenate((otherFeatures, headLineVec, prosVec, consVec), axis=0)).astype('float32')

    updatedDataFrame.append(finalArray)

datasetLabels = dataset["overall_rating"].apply(lambda x: x-1)
dataset = dataset.apply(lambda x: dataFrameToNpArray(x), axis=1)
dataset = np.asarray(updatedDataFrame)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


# Usage example:

data = {
    "Training Data (Actual)": [0, 0, 0, 0, 0, 0],
    "Training Data (Predicted)": [0, 0, 0, 0, 0, 0],
    "Test Data (Predicted)": [0, 0, 0, 0, 0, 0],
}

for idx in datasetLabels:
    data["Training Data (Actual)"][1+ int(idx)] = data["Training Data (Actual)"][1+int(idx)] + 1/len(datasetLabels)

model = models.Sequential()

model.add(layers.Dense(1024, activation='relu', input_shape=(905,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(.1))
model.add(layers.Dense(5))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

predictions = model.predict(dataset)

updatedPrediction = []
for i in range(0, len(predictions)):
    updatedPrediction.append(np.argmax(predictions[i]))

for idx in updatedPrediction:
    data["Training Data (Predicted)"][1+ int(idx)] = data["Training Data (Predicted)"][1+int(idx)] + 1/len(updatedPrediction)


testingSet = pd.read_csv("Econ424_F2023_PC6_glassdoor_training_large_v1.csv")
testingSet.drop(columns=["location", "overall_rating"], inplace=True)

def isInFirmDict(x):
    if x in firmDict:
        return firmDict[x]/ len(firmDict)
    else:
        return 0

def isInJobTitleDict(x):
    if x in jobTitleDict:
        return jobTitleDict[x]/ len(jobTitleDict)
    else:
        return 0

testingSet["firm"] = testingSet['firm'].apply(lambda x: isInFirmDict(x))
testingSet["job_title"] = testingSet['job_title'].apply(lambda x: isInJobTitleDict(x))
testingSet["date_review"] = testingSet['date_review'].apply(lambda x: convertDayTime(x) / 2023119)
testingSet["year"] = testingSet['year'].apply(lambda x: x / 2023)

testingSet["pros"] = testingSet["pros"].apply(lambda x: cleanAllStringsAndAdd(x))
testingSet["headline"] = testingSet["headline"].apply(lambda x: cleanAllStringsAndAdd(x))
testingSet["cons"] = testingSet["cons"].apply(lambda x: cleanAllStringsAndAdd(x))

updatedDataFrame = []

testingSet = testingSet.apply(lambda x: dataFrameToNpArray(x), axis=1)
testingSet = np.asarray(updatedDataFrame)

predictions = model.predict(testingSet)

updatedPrediction = []
for i in range(0, len(predictions)):
    updatedPrediction.append(np.argmax(predictions[i]))

for idx in updatedPrediction:
    data["Test Data (Predicted)"][1+ int(idx)] = data["Test Data (Predicted)"][1+int(idx)] + 1/len(updatedPrediction)

fig, ax = plt.subplots()
bar_plot(ax, data, total_width=.8, single_width=.9)

plt.show()