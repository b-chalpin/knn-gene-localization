import pandas as pd
import numpy as np
import preprocessing
import classifier

train_data = preprocessing.load_train_dataset("./data/Genes_relation.data")
test_data = preprocessing.load_test_dataset("./data/Genes_relation.test")

print(train_data["Complex"].value_counts())
print("NAN Count:", train_data["Complex"].isna().count())

# print(train_data.head(5))

# prediction = classifier.knn_classifier(train_data, test_data, 3)
