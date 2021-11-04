import preprocessing
import classifier

# load training and test data
train_data = preprocessing.load_train_dataset("./data/Genes_relation.data")
test_data = preprocessing.load_test_dataset("./data/Genes_relation.test")

test_data = test_data.sample(10, random_state=15785, replace=False, ignore_index=True)
# train_data = train_data.sample(1000, random_state=11234, replace=False, ignore_index=True)

# use KNN with n=3
prediction = classifier.knn_classifier(train_data, test_data, 3)

# validate prediction
accuracy = classifier.calculate_accuracy(prediction, test_data)
print(f"\n\nAccuracy of KNN model: {accuracy*100}%")
