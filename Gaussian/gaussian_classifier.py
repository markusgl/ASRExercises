import pandas as pd
import numpy as np
import datetime
import time

from math import log, pi, pow

# Set time stamp for log file
ts = time.time()
ts_formatted = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Setup parameters
training_examples_limit = None  # 'None' for training on all examples
test_examples_limit = None  # 'None' for testing on all examples
training_data_file = '../data/TIMIT/train.mfcccsv'
training_labels_file = '../data/TIMIT/train.targphon'
test_data_file = '../data/TIMIT/test.mfcccsv'
test_labels_file = '../data/TIMIT/test.targphon'
logfile = open('gaussian_classifier_results.log', 'a')
derivatives = True

# Read training data
df_train = pd.read_csv(training_data_file, sep=',', nrows=training_examples_limit)
np_train = df_train.values

with open(training_labels_file, 'r') as f:
    train_labels = []
    for row in f.readlines():
        train_labels.append(row)


# Read test data
df_test = pd.read_csv(test_data_file, sep=',', nrows=test_examples_limit)
np_test = df_test.values

with open(test_labels_file, 'r') as f:
    test_labels = []
    for row in f.readlines():
        test_labels.append(row)

def extract_with_derivatives(): 
    derivatives_training = get_dimensional_vector(np_train)
    derivatives_test     = get_dimensional_vector(np_test)
    return (derivatives_training, train_labels), (derivatives_test, test_labels)

def get_dimensional_vector(feature_vector):
    first = np.array([feature_vector[0]])
    last  = np.array([feature_vector[len(feature_vector)-1]])
    copy_neighbors = np.append(first, feature_vector, axis=0)
    copy_neighbors = np.append(copy_neighbors, last, axis=0)
    neighbors = np.array([])
    for i, value in enumerate(feature_vector):
        if(len(neighbors)): #if it is not empty#
            neighbors = np.append(neighbors, calculate_derivatives(copy_neighbors[i], copy_neighbors[i+1], copy_neighbors[i+2]), axis=0)
        else: 
            neighbors = calculate_derivatives(copy_neighbors[i], copy_neighbors[i+1], copy_neighbors[i+2])
    return neighbors

def calculate_derivatives(previous, current, following):
    delta = np.subtract(following, previous) / 2
    delta_delta = np.add(np.subtract(previous, 2*current), following)
    return np.array([np.append(current, np.append(delta, delta_delta))])

if(derivatives):
    (np_train, train_labels), (np_test, test_labels) = extract_with_derivatives()
# Assign each training example to its class label
class_features = {}
class_example_counts = {}
for i, vector in enumerate(np_train):
    if train_labels[i] in class_features.keys():
        class_features[train_labels[i]] = np.concatenate((class_features[train_labels[i]], np.array([vector])), axis=0)
        class_example_counts[train_labels[i]] += 1
    else:
        class_features[train_labels[i]] = np.array([vector])
        class_example_counts[train_labels[i]] = 1

# Calculate means, variance and priori probability for each class
means_dict = {}
variance_dict = {}
priori = {}
for key, value in class_features.items():
    means_dict[key] = np.mean(value, axis=0)
    variance_dict[key] = np.var(value, axis=0)
    priori[key] = class_example_counts[key] / len(np_train)


def gaussian_classify():
    error_count = 0
    gauss_dists = {}

    for i, test_vector in enumerate(np_test):
        test_label = test_labels[i]

        for class_label, features_vector in class_features.items():
            sum_variances = 0
            sum_means = 0
            gauss_dist = -2 * log(priori[class_label])
            for j in range(len(test_vector)):
                sum_variances += log(2 * pi * variance_dict[class_label][j])
                sum_means += pow(test_vector[j] - means_dict[class_label][j], 2) / variance_dict[class_label][j]

            gauss_dist += sum_variances + sum_means
            gauss_dists[class_label] = gauss_dist

        min_prob = min(gauss_dists, key=gauss_dists.get)

        if not min_prob == test_label:
            error_count += 1

    recognition_rate = 1 - error_count / len(np_test)
    print(f'Recognition rate is {recognition_rate}')
    if training_examples_limit and test_examples_limit:
        logfile.write(
            f'Recognition rate using {training_examples_limit} training examples and {test_examples_limit} test examples is '
            f'{recognition_rate}')
    else:
        logfile.write(
            f'Recognition rate is {recognition_rate}')


logfile.write(f'{ts_formatted} Starting Gaussian classification:')
gaussian_classify()

