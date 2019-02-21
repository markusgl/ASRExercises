import pandas as pd
import numpy as np
import operator
import datetime
import time

from math import sqrt

# set time stamp for log file
ts = time.time()
ts_formatted = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# setup parameters
training_examples = 5000
test_examples = 1000
k_start = 1
k_end = 25
logfile = open('knn_normalized_results.log', 'a')
derivatives = True

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

# Read training data
df_train = pd.read_csv('../data/TIMIT/train.mfcccsv', sep=',', nrows=training_examples)
np_train = df_train.values

with open('../data/TIMIT/train.targphon', 'r') as f:
    train_labels = []
    for row in f.readlines():
        train_labels.append(row)

# Read test data
df_test = pd.read_csv('../data/TIMIT/test.mfcccsv', sep=',', nrows=test_examples)
np_test = df_test.values

with open('../data/TIMIT/test.targphon', 'r') as f:
    test_labels = []
    for row in f.readlines():
        test_labels.append(row)

if(derivatives):
    (np_train, train_labels), (np_test, test_labels) = extract_with_derivatives()

# calculate means and standard deviation for train and test features
means_train = np_train.sum(axis=0) / len(np_train)
standard_deviation_train = np_train.std(axis=0)
means_test = np_test.sum(axis=0) / len(np_test)
standard_deviation_test = np_test.std(axis=0)


def k_nearest(k):
    error_count = 0
    for i, test_vector in enumerate(np_test):
        shortest_distances = []
        target_labels = []
        test_vector = (test_vector - means_test) / standard_deviation_test

        for j, train_vector in enumerate(np_train):
            # normalize vectors
            train_vector = (train_vector - means_train) / standard_deviation_train

            # measure distance
            distance = sqrt(sum(np.power((train_vector - test_vector), 2)))

            # fill target labels to length of k
            if len(shortest_distances) < k:
                shortest_distances.append(distance)
                target_labels.append(train_labels[j])
            # compare current distances to max value in list
            elif distance < max(shortest_distances):
                max_value = max(shortest_distances)
                max_index = shortest_distances.index(max_value)
                shortest_distances[max_index] = distance
                target_labels[max_index] = train_labels[j]

        # choose final label
        label_count = {}
        for label in target_labels:
            if not label in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        target_label = max(label_count.items(), key=operator.itemgetter(1))[0]
        test_label = test_labels[i]
        if not target_label == test_label:
            error_count += 1
        #print(f'Target label: {target_label}')
        #print(f'Actual label: {test_label}')
        #print(f'Shortest distance: {shortest_distance}')

    recognition_rate = 1 - error_count / len(np_test)
    print(f'Recognition rate for k {k} is {recognition_rate}')
    logfile.write(f'Recognition rate for k {k} is {recognition_rate}\n')



# test different values for k
print(f'Testing k values from {k_start} to {k_end} for {training_examples} training examples and '
      f'{test_examples} test examples%s:' % (" with derivatives" if derivatives else ""))
logfile.write(f'{ts_formatted} Testing k values from {k_start} to {k_end} for {training_examples} training examples '
              f'{test_examples} test examples:\n')

for i in range(k_start, k_end + 1):
    k_nearest(i)

