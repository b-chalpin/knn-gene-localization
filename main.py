import pandas as pd
import numpy as np
import preprocessing
import classifier

train_data = preprocessing.load_csv("./data/Genes_relation.data")
test_data = preprocessing.load_csv("./data/Genes_relation.test")

train_data = preprocessing.encode_categorical_variables(train_data)
print(train_data.columns)

# print(train_data.head(5))

# prediction = classifier.knn_classifier(train_data, test_data, 3)
