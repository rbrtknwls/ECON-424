import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
import re
import gensim


from tensorflow.keras import layers, models
from gensim.models import KeyedVectors
from sklearn import metrics

dataset = pd.read_csv("Econ424_F2023_PC6_glassdoor_training_large_v1.csv")

firmDict = {}
jobTitleDict = {}
sentences = []


def convertString(string):
    return (''.join(ch for ch in string if ch.isalnum())).lower()


def updateDicts(row):
    firm = row["firm"]
    job = row["job_title"]

    firm = convertString(firm)
    job = convertString(job)

    if firm not in firmDict:
        firmDict[firm] = len(firmDict) + 1

    if job not in jobTitleDict:
        jobTitleDict[job] = len(jobTitleDict) + 1


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
#dataset = dataset.apply(lambda x: dataFrameToNpArray(x), axis=1)
#dataset = np.asarray(updatedDataFrame)

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

# ================== TESTING ============================

print(model.summary())


model.load_weights("modelAtEpoch7.keras")

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

f = open('predictions.csv', 'w')
for estimate in updatedPrediction:
    f.writelines(str(estimate+1) + ",\n")
print(updatedPrediction)