import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    ''' 
        Method to load in data from csv file path 

        - Replaces "?" with NaN
        - Drops the "Function" attribute
    '''
    data = pd.read_csv(path, dtype=dict(GeneID="string", Essential="string", Class="string",
                                        Complex="string", Phenotype="string", Motif="string",
                                        Chromosome="string", Function="string", Localization="string"))

    # # Replace ? with NaN (Null)
    data.replace("?", np.NaN, inplace=True)

    # Remove Function attribute
    data = data.drop(["Function"], axis=1)

    return data


def load_train_dataset(path: str) -> pd.DataFrame:
    '''
        Load and preprocess our train dataset
    '''
    return load_csv(path)


def load_test_dataset(path: str) -> pd.DataFrame:
    '''
        Load and preprocess our test dataset
        
        - Load the Genes_relation.test file
        - Load the keys.txt file
        - Join the keys and test data on "GeneID"
    '''
    # load raw test dataset
    raw_test = load_csv(path)

    # drop Localization attribute in test data
    raw_test = raw_test.drop(["Localization"], axis=1)

    # load the keys.txt data
    keys = pd.read_csv("./data/keys.txt", dtype=dict(GeneID="string", Localization="string"))

    # join keys and test data on GeneID
    return pd.merge(raw_test, keys, on="GeneID", how="inner")
