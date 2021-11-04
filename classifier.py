import pandas as pd
import numpy as np


'''
   Distance function for calculating euclidean distance between two tuples 
'''
def calculate_distance(u, v) -> float:
    distance = 0

    for index in range(2, len(u) - 1):
        # add 0.5 to distance if there is a missing value
        if pd.isna(u[index]) or pd.isna(v[index]):
            distance += 0.5

        else:
            # if attributes do not match, add 1 to distance
            if u[index] != v[index]:
                distance += 1

    return distance


'''
    Main KNN function
'''
def knn_classifier(train: pd.DataFrame, test: pd.DataFrame, n: int) -> np.array:
    # array of test data predictions to return
    prediction = []

    # for each record in test
    for test_row in test.itertuples():
        # initialize dataframe to hold distance to neighbor information
        neighbors = pd.DataFrame(columns=["Distance", "Localization"])

        print("Begin K-Nearest Neighbor classification for test tuple:\n", test_row)

        # for each training tuple
        for train_row in train.itertuples():
            # calculate distance measure
            distance = calculate_distance(test_row, train_row)

            # add the distance and the class of the neighbor to the "neighbors" list
            neighbors = neighbors.append({"Distance": distance, "Localization": train_row[8]},
                                         ignore_index=True)

        # sort neighbors list by distance ascending and fetch the "n-nearest neighbors"
        nearest_neighbors = neighbors.sort_values(by=["Distance"], ascending=True)
        n_nearest_neighbors = nearest_neighbors[0:n]

        # get the majority vote -- return the Localization value with the most votes
        majority_vote = n_nearest_neighbors["Localization"].value_counts().idxmax()

        # store the prediction
        prediction = np.append(prediction, np.array([majority_vote]), axis=0)

        print("Prediction complete for test tuple. Prediction: ", majority_vote)

    return np.array(prediction)


'''
    Performance calculation
'''
def calculate_accuracy(predictions: np.array, test: pd.DataFrame) -> float:
    correct_predictions = 0

    # convert dataframe to array
    test_array = test.to_numpy()

    for pred, test in zip(predictions, test_array):
        # if equal, add 1 to numerator
        if pred == test[7]:
            correct_predictions += 1

    # numerator / total test tuples = accuracy
    accuracy = correct_predictions / predictions.size

    return accuracy
