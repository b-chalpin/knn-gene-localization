import pandas as pd
import numpy as np


''' 
    Method to load in data from csv file path 
    
      - Replaces "?" with NaN
      - Drops the "Function" attribute
'''
def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.replace("?", np.nan)
    return data.drop(["Function"], axis=1)


''' 
    Replace missing values with the average of each attribute 
'''
def replace_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

    return data


data = load_dataset("./data/Genes_relation.data")
print(data.head(10))
# data = replace_missing_values(data)
# print(data.head(10))

names = pd.read_csv("./data/Genes_relation.names", sep="\n")
print(names)
