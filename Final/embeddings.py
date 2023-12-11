from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

testingSet = pd.read_csv("Econ424_F2023_PC6_glassdoor_training_large_v1.csv")

sentences = []

def addToSentence(row):
    if type(row["firm"]) == str:
        sentences.append(row["firm"])
    if type(row["job_title"]) == str:
        sentences.append(row["job_title"])
    if type(row["headline"]) == str:
        sentences.append(row["headline"])
    if type(row["pros"]) == str:
        sentences.append(row["pros"])
    if type(row["cons"]) == str:
        sentences.append(row["cons"])


testingSet.apply(lambda row: addToSentence(row), axis=1)

embeddings = model.encode(sentences, show_progress_bar=True)

# Store sentences & embeddings on disc
with open('embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
