import pandas as pd
import numpy as np
from output import Logger, ResultFileWriter


def calculate_distance(u, v) -> float:
    '''
        Distance function for calculating euclidean distance between two tuples 
    '''
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


def knn_classifier(train: pd.DataFrame, test: pd.DataFrame, n: int) -> np.array:
    '''
        Main KNN function
    '''
    # initialize logger
    logger = Logger()

    # initialize file writer
    results_file_writer = ResultFileWriter()

    print(f"Beginning KNN classification for k = {n}")
    logger.log(f"Beginning KNN classification for train set {train.shape} and test set {test.shape}\n\n")

    # array of test data predictions to return
    prediction_list = []

    # convert dataframes to NumPy arrays -- this makes iterating through them faster
    train_data = train.to_numpy()
    test_data = test.to_numpy()

    for test_tuple in test_data:
        # initialize dataframe to hold distance to neighbor information
        neighbors = pd.DataFrame(columns=["Distance", "Localization"])

        print(f"Begin K-Nearest Neighbor classification for test tuple:\n{test_tuple}")
        logger.log(f"Begin classification for test tuple:\n{test_tuple}")

        # for each training tuple
        for train_row in train_data:
            # calculate distance measure
            distance = calculate_distance(test_tuple, train_row)

            # add the distance and the Localization of the neighbor to the "neighbors" list
            neighbors = neighbors.append({"Distance": distance, "Localization": train_row[7]},
                                         ignore_index=True)

        # get n neighbors with the smallest Distance value
        n_nearest_neighbors = neighbors.nsmallest(n, columns="Distance")

        # get the majority vote -- return the Localization value with the most votes
        majority_vote = n_nearest_neighbors["Localization"].value_counts().idxmax()

        # store the prediction for validation later
        prediction_list = np.append(prediction_list, np.array([majority_vote]), axis=0)

        # store prediction result in results file
        results_file_writer.store_prediction_result(test_tuple[0], majority_vote)

        print(f"Prediction complete for test tuple. Prediction: {majority_vote}")
        logger.log(f"PREDICTION: {majority_vote}")

    return np.array(prediction_list)


def calculate_accuracy(predictions: np.array, test: pd.DataFrame) -> float:
    '''
        Performance calculation
    '''
    # initialize logger
    logger = Logger()

    correct_predictions = 0

    # convert dataframe to array
    test_array = test.to_numpy()

    for pred, test in zip(predictions, test_array):
        # index 7 in test is "Localization"
        if pred == test[7]:
            correct_predictions += 1

    accuracy = correct_predictions / predictions.size

    logger.log(f"Overall accuracy of KNN model: {accuracy * 100}%")

    return accuracy
