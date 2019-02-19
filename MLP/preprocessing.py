import numpy as np
import pandas as pd


class Preprocessor():
    def __init__(self):
        # Setup parameters
        self.training_examples_limit = 5000  # 'None' for training on all examples
        self.test_examples_limit = 200  # 'None' for testing on all examples
        self.training_data_file = '../data/TIMIT/train.mfcccsv'
        self.training_labels_file = '../data/TIMIT/train.targphon'
        self.test_data_file = '../data/TIMIT/test.mfcccsv'
        self.test_labels_file = '../data/TIMIT/test.targphon'

        self.class_labels_to_int = {}
        #self.logfile = open('gaussian_classifier_results.log', 'a')

    def read_training_test_data(self):
        # Read training data
        df_train = pd.read_csv(self.training_data_file, sep=',', nrows=self.training_examples_limit)
        np_train = df_train.values

        with open(self.training_labels_file, 'r') as f:
            train_labels = []
            for row in f.readlines():
                train_labels.append(row)

        # Read test data
        df_test = pd.read_csv(self.test_data_file, sep=',', nrows=self.test_examples_limit)
        np_test = df_test.values

        with open(self.test_labels_file, 'r') as f:
            test_labels = []
            for row in f.readlines():
                test_labels.append(row)

        return np_train, train_labels, np_test, test_labels

    def create_class_labels_dict(self, train_labels):
        count = 0
        for label in train_labels:
            if not label in self.class_labels_to_int.keys():
                self.class_labels_to_int[label] = count
                count += 1

    def class_to_int(self, class_label):
        return self.class_labels_to_int[class_label]

    def int_to_class(self, class_to_int_dict, int):
        for key, value in self.class_labels_to_int.items():
            if value == int:
                return key

    def extract_features(self):
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        self.create_class_labels_dict(train_labels)

        y_train = np.zeros(shape=(0, len(train_labels)))
        y_test = np.zeros(shape=(0, len(test_labels)))
        for label in train_labels:
            y_train = np.append(y_train, self.class_to_int(label))
        for label in test_labels:
            y_test = np.append(y_test, self.class_to_int(label))

        return (np_train, y_train), (np_test, y_test)


if __name__ == '__main__':
    prep = Preprocessor()
    (x_train, y_train), (x_test, y_test) = prep.extract_features()
    print(len(x_test))
