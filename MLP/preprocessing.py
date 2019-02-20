import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self):
        # Setup parameters

        self.training_examples_limit = 100000  # 'None' for training on all examples
        self.test_examples_limit = 10000  # 'None' for testing on all examples
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

    def neighbor_n_features(self, feature_vector, n):
        firsts = np.array([feature_vector[0], feature_vector[0]])
        lasts  = np.array([feature_vector[len(feature_vector)-1], feature_vector[len(feature_vector)-1]])
        copy_neighbors = np.append(firsts, feature_vector, axis=0)
        copy_neighbors = np.append(copy_neighbors, lasts, axis=0)
        neighbors = np.array([])
        for i, value in enumerate(feature_vector):
            if(len(neighbors)): #if it is not empty#
                neighbors = np.append(neighbors, np.array([copy_neighbors[i:i+n]]), axis=0)
            else: 
                neighbors = np.array([copy_neighbors[i:i+n]])
        #should return 3D array
        return neighbors

    def extract_with_neighbor_features(self):
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        neighbors_amount = 5
        neighbors_samples = self.neighbor_n_features(np_train, neighbors_amount)
        neighbors_test = self.neighbor_n_features(np_test, neighbors_amount)
        return (neighbors_samples, train_labels), (neighbors_test, test_labels)

    def extract_multi_feature_vectors(self):
        """
        Approach by Markus
        :return:
        """
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        mult_features = []
        mult_vector = []
        for i, feature in enumerate(np_train):
            if i == 0:
                mult_vector = np.vstack((np_train[i:i + 1], np_train[i:i + 1], np_train[i:i + 1],
                                         np_train[i + 1:i + 2], np_train[i + 2:i + 3]))
            elif i == 1:
                mult_vector = np.vstack((np_train[i:i + 1], np_train[:i], np_train[i:i + 1],
                                         np_train[i + 1:i + 2], np_train[i + 2:i + 3]))
            elif i == len(np_train) - 2:
                mult_vector = np.vstack(
                    (np_train[i - 2:i - 1], np_train[i - 1:i], np_train[i:i + 1], np_train[i + 1:i + 2],
                     np_train[i:i + 1]))
            elif i == len(np_train) - 1:  # last element
                mult_vector = np.vstack(
                    (np_train[i - 2:i - 1], np_train[i - 1:i], np_train[i:i + 1], np_train[i:i + 1],
                     np_train[i:i + 1]))
            elif 2 < i < len(np_train)-3:
                mult_vector = np_train[i - 2:i + 3]

            size = mult_vector.shape[0] * mult_vector.shape[1]
            reshaped_vector = np.reshape(mult_vector, size)
            mult_features.append(reshaped_vector)

        return (mult_features, train_labels), (np_test, test_labels)
    
    def extract_with_derivatives(self): 
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        derivatives_training = self.get_dimensional_vector(np_train)
        derivatives_test     = self.get_dimensional_vector(np_test)

        np.set_printoptions(threshold=np.nan)
        return (derivatives_training, train_labels), (derivatives_test, test_labels)

    def get_dimensional_vector(self, feature_vector):
        first = np.array([feature_vector[0]])
        last  = np.array([feature_vector[len(feature_vector)-1]])
        copy_neighbors = np.append(first, feature_vector, axis=0)
        copy_neighbors = np.append(copy_neighbors, last, axis=0)
        neighbors = np.array([])
        for i, value in enumerate(feature_vector):
            if(len(neighbors)): #if it is not empty#
                neighbors = np.append(neighbors, self.calculate_derivatives(copy_neighbors[i], copy_neighbors[i+1], copy_neighbors[i+2]), axis=0)
            else: 
                neighbors = self.calculate_derivatives(copy_neighbors[i], copy_neighbors[i+1], copy_neighbors[i+2])
        return neighbors

    def calculate_derivatives(self, previous, current, following):
        delta = np.add(previous, following) / 2
        delta_delta = np.add(np.subtract(previous, 2*current), following)
        return np.array([np.append(current, np.append(delta, delta_delta))])

    def calc_V_vector(self):
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        np_v_train = np.array()
        np_v_test = np.array()
        for i, feature in enumerate(np_train):
            if (i - 1) < len(np_train) and (i > 0) :
                np_v_train[i] = (np_train[i - 1] - np_train[i + 1]) / 2
            elif i == 0:
                np_v_train[i] = (np_train[i] - np_train[i + 1]) / 2
            elif i == len(np_train):
                np_v_train[i] = (np_train[i - 1] - np_train[i]) / 2
        for i, feature in enumerate(np_test):
            if (i - 1) < len(np_test) and (i > 0) :
                np_v_test[i] = (np_test[i - 1] - np_test[i + 1]) / 2
            elif i == 0:
                np_v_test[i] = (np_test[i] - np_test[i + 1]) / 2
            elif i == len(np_test):
                np_v_test[i] = (np_test[i - 1] - np_test[i]) / 2

    def calc_A_vector(self):
        np_train, train_labels, np_test, test_labels = self.read_training_test_data()
        np_a_train = np.array()
        np_a_test = np.array()
        for i, feature in enumerate(np_train):
            if (i - 1) < len(np_train) and (i > 0) :
                np_v_train[i] = np_train[i - 1] - 2 * np_train[i] + np_train[i + 1]
            elif i == 0:
                np_v_train[i] = np_train[i] - 2 * np_train[i] + np_train[i + 1]
            elif i == len(np_train):
                np_v_train[i] = np_train[i - 1] - 2 * np_train[i] + np_train[i]
        for i, feature in enumerate(np_test):
            if (i - 1) < len(np_test) and (i > 0) :
                np_a_test[i] = np_test[i - 1] - 2 * np_test[i] + np_test[i + 1]
            elif i == 0:
                np_a_test[i] = np_test[i] - 2 * np_test[i] + np_test[i + 1]
            elif i == len(np_test):
                np_a_test[i] = np_test[i - 1] - 2 * np_test[i] + np_test[i]

    def merge_feature_vectors(self):
        return None


if __name__ == '__main__':
    prep = Preprocessor()

    prep.extract_with_neighbor_features()

