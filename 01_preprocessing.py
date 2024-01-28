import pandas as pd
import os

df = pd.read_csv("data/Cardiovascular_Disease_Dataset.csv")

print(df.head())
df.drop("patientid", axis=1, inplace=True)

DIR="data/preprocessed"
os.makedirs(DIR, exist_ok=True)
df.to_csv(DIR+"/preprocessed_data.csv", index=False)


