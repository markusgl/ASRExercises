import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        # Setup parameters
        self.training_examples_limit = None  # 'None' for training on all examples
        self.test_examples_limit = None  # 'None' for testing on all examples
        self.training_data_file = '../data/TIMIT/train.mfcccsv'
        self.training_labels_file = '../data/TIMIT/train.targcsv'
        self.test_data_file = '../data/TIMIT/test.mfcccsv'
        self.test_labels_file = '../data/TIMIT/test.targcsv'

        self.class_labels_to_int = {}

    def read_training_test_data(self):
        # Read training data
        df_train = pd.read_csv(self.training_data_file, sep=',', nrows=self.training_examples_limit)
        np_train = df_train.values

        df_train_labels = pd.read_csv(self.training_labels_file, sep=',', nrows=self.training_examples_limit)
        train_labels = df_train_labels.values

        # Read test data
        df_test = pd.read_csv(self.test_data_file, sep=',', nrows=self.test_examples_limit)
        np_test = df_test.values

        df_test = pd.read_csv(self.test_labels_file, sep=',', nrows=self.test_examples_limit)
        test_labels = df_test.values

        return np_train, train_labels, np_test, test_labels

    def extract_features(self):
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()

        return (np_train, train_labels), (np_test, test_labels)


    def extract_with_neighbor_features(self):
        # TODO
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()

        first_index = np_train[0]
        last_index = np_train[len(np_train)-1]
        copy = np.append([first_index], [first_index], np_train, [last_index], [last_index])
        feature5vector = []

        for i, row in enumerate(np_train):
            """
            if i == 0:
                np_neighbor = row + np_train[i+1] + np_train[i+2]
            elif i < 2:
                np_neighbor = np.concatenate(row + np_train[i+1] + np_train[i+2])
            elif i > len(np_train) - 2:
                np_neighbor = np.concatenate(row + np_train[i - 1] + np_train[i - 2])
            else:
                np_neighbor = np.concatenate(row + np_train[i - 1] + np_train[i - 2])
            """

            feature5vector = np.concatenate(feature5vector, copy[i:i+5])


if __name__ == '__main__':
    prep = Preprocessor()
    prep.extract_with_neighbor_features()
