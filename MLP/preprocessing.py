import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        # Setup parameters
        self.training_examples_limit = 100000  # 'None' for training on all examples
        self.test_examples_limit = 1000  # 'None' for testing on all examples
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
        #n_labels_train = self.neighbor_n_features(train_labels, neighbors_amount)
        neighbors_test = self.neighbor_n_features(np_test, neighbors_amount)
        #n_labels_test = self.neighbor_n_features(test_labels, neighbors_amount)
        return (neighbors_samples, train_labels), (neighbors_test, test_labels)

    
if __name__ == '__main__':
    prep = Preprocessor()
    prep.extract_with_neighbor_features()
