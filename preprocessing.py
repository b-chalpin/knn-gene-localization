import pandas as pd
import numpy as np


''' 
    Method to load in data from csv file path 

      - Replaces "?" with NaN
      - Drops the "Function" attribute
'''
def load_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    # Replace ? with NaN (Null)
    data[data == '?'] = np.NaN

    # Remove Function attribute
    data = data.drop(["Function"], axis=1)

    # Replace missing values with column means
    for col in data.columns.drop(["GeneID", "Localization"]):
        data[col] = data[col].fillna(data[col].mode()[0], inplace=True)

    return data


''' 
    Replace missing values with the average of each attribute 
'''
def replace_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns.drop(["GeneID", "Localization"]):
        mode = data[col].mode()[0]
        data[col] = data[col].fillna(data[col].mode()[0], inplace=True)

    return data


'''
    Load and preprocess our train dataset
'''
def load_train_dataset(path: str) -> pd.DataFrame:
    test_data = load_csv(path)
    # return replace_missing_values(test_data)


'''
    Load and preprocess our test dataset
'''
def load_test_dataset(path: str) -> pd.DataFrame:
    return load_csv(path)
    # return replace_missing_values(test_data)
