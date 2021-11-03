import pandas as pd
import numpy as np
from scipy.spatial import distance


'''
   Distance function for calculating euclidean distance between two tuples 
'''
def calculate_cosine_similarity(u: pd.Series, v: pd.Series) -> float:
    return 1 - distance.cosine(u, v)


'''
    Main KNN function
'''
def knn_classifier(train: pd.DataFrame, test: pd.DataFrame, n: int) -> np.array:
    # array of test data predictions to return
    prediction = np.zeros(len(test.index))

    # for each record in test
    for test_index, test_row in test.iterrows():
        for train_index, train_row in train.iterrows():
            if test_index == 1 and train_index < 5:
                print(calculate_cosine_similarity(test_row, train_row))

        # calculate distance between test and all training

        # sort in ascending of distance

        # pick first n sorted distance tuples

        # test prediction is the majority vote of Localization for the n picked tuples

    return prediction

