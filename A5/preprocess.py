import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import warnings
from feature_engine.encoding import RareLabelEncoder

warnings.filterwarnings("ignore")

# Read in the data
data = pd.read_csv("Econ424_F2023_PC4_training_data_large.csv", low_memory=False)

# ================ General Data Parsing ================
toNumeric = ["bed_length", "back_legroom", "front_legroom", "height", "length", "wheelbase", "width", "maximum_seating",
             "fuel_tank_volume"]


data.drop(columns=["major_options", "power", "exterior_color", "city", "interior_color", "listing_color",
                   "transmission", "bed", "bed_height"], inplace=True)

def parseIntoNumeric(x):
    if type(x) == str:
        return float(x.split(" ")[0])
    else:
        return np.nan


for col in data.columns:
    data[col].replace('--', np.nan, inplace=True)
    if col in toNumeric:
        print(col)
        data[col] = data[col].apply(lambda x: parseIntoNumeric(x))



catagories = ["body_type", "engine_type", "fleet", "frame_damaged", "franchise_dealer", "franchise_make",
              "fuel_type", "has_accidents", "iscab", "is_certified", "is_cpo", "is_new", "is_oemcpo",
              "make_name", "model_name", "salvage", "sp_name", "theft_title", "transmission_display",
              "trimid", "trim_name", "vehicle_damage_category", "wheel_system", "wheel_system_display"]

data.fillna(0, inplace=True)

for col in data.columns:
    if col in catagories:
        print(data.dtypes[col])
        data[col] = data[col].fillna('None').astype(str)
        encoder = RareLabelEncoder(n_categories=1, max_n_categories=2500, replace_with='Other', tol=40 / data.shape[0])
        data[col] = encoder.fit_transform(data[[col]])


print(data.dtypes)
print(data.head(4))
data.to_csv("processed.csv", index=False)
