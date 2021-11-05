import time
import preprocessing
import classifier


def main():
    # load training and test data
    train_data = preprocessing.load_train_dataset("./data/Genes_relation.data")
    test_data = preprocessing.load_test_dataset("./data/Genes_relation.test")

    # uncomment the next two next lines to under-sample our test set and train set for faster performance
    #   - test dataset will be reduced to 300 tuples randomly sampled without replacement
    #   - train dataset will be reduced to 300 tuples randomly sampled without replacement
    # test_data = test_data.sample(300, random_state=15785, replace=False, ignore_index=True)   # UNCOMMENT THIS LINE
    # train_data = train_data.sample(300, random_state=11234, replace=False, ignore_index=True) # UNCOMMENT THIS LINE

    # predict Localization for our test data using KNN with k = 3
    before_knn = time.time()  # time logging
    prediction = classifier.knn_classifier(train_data, test_data, 3)
    after_knn = time.time()  # time logging

    # validate prediction
    accuracy = classifier.calculate_accuracy(prediction, test_data)
    print(f"\n\nAccuracy of KNN model: {accuracy * 100}%\n"
          f"Time elapsed: {after_knn - before_knn} sec")


main()
