import pandas as pd
import sys
from NaiveBayesClassifier import Classifier

train_file_name = ""
test_file_name = ""

if __name__ == '__main__':
    # sys.argv[0] is the name of the program, we can skip it
    path_to_train_file = sys.argv[1]  # this takes the 1st terminal argument
    path_to_test_file = sys.argv[2]  # this takes the 2nd terminal argument

train_data = pd.read_table(path_to_train_file, sep="\t")
test_data = pd.read_table(path_to_test_file, sep="\t")

classifier = Classifier(train_data)
classifier.build_classifiers()

classifier.print_str()

classifier.train_accuracy()
classifier.test_accuracy(test_df=test_data)
