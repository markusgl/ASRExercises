import pandas as pd
import numpy as np
import operator

from math import sqrt

# setup parameters
training_examples = 2000
test_examples = 500
k_start = 1
k_end = 10


# Read training data
df_train = pd.read_csv('data/TIMIT/train.mfcccsv', sep=',', nrows=training_examples)
np_train = df_train.values

with open('data/TIMIT/train.targphon', 'r') as f:
    train_labels = []
    for row in f.readlines():
        train_labels.append(row)

# Read test data
df_test = pd.read_csv('data/TIMIT/test.mfcccsv', sep=',', nrows=test_examples)
np_test = df_test.values

with open('data/TIMIT/test.targphon', 'r') as f:
    test_labels = []
    for row in f.readlines():
        test_labels.append(row)


def k_nearest(k):
    error_count = 0
    for i, test_vector in enumerate(np_test):
        shortest_distances = []
        target_labels = []

        for j, train_vector in enumerate(np_train):
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


# test different values for k
print(f'Testing k values from {k_start} to {k_end} for {training_examples} training examples '
      f'{test_examples} test examples:')
for i in range(k_start, k_end + 1):
    k_nearest(i)
