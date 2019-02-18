import pandas as pd
import numpy as np
import datetime
import time

from math import log, pi, pow

# set time stamp for log file
ts = time.time()
ts_formatted = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# setup parameters
training_examples = 100000
test_examples = 2000

logfile = open('knn_normalized_results.log', 'a')

# Read training data
df_train = pd.read_csv('../KNN/data/TIMIT/train.mfcccsv', sep=',', nrows=training_examples)
np_train = df_train.values

with open('../KNN/data/TIMIT/train.targphon', 'r') as f:
    train_labels = []
    for row in f.readlines():
        train_labels.append(row)

# Read test data
df_test = pd.read_csv('../KNN/data/TIMIT/test.mfcccsv', sep=',', nrows=test_examples)
np_test = df_test.values

with open('../KNN/data/TIMIT/test.targphon', 'r') as f:
    test_labels = []
    for row in f.readlines():
        test_labels.append(row)


# calculate means and standard deviation for each class of the training data
sum_dict = {}
count_classes = {}
concat = []
for i, vector in enumerate(np_train):
    if train_labels[i] in sum_dict.keys():
        sum_dict[train_labels[i]] = np.concatenate((sum_dict[train_labels[i]], np.array([vector])), axis=0)
        count_classes[train_labels[i]] += 1
    else:
        sum_dict[train_labels[i]] = np.array([vector])
        count_classes[train_labels[i]] = 1


means_dict = {}
variance_dict = {}
priori = {}
for key, value in sum_dict.items():
    means_dict[key] = np.mean(value, axis=0)
    variance_dict[key] = np.var(value, axis=0)
    priori[key] = count_classes[key] / len(np_train)


def gaussian():
    error_count = 0
    probs = {}

    for i, test_vector in enumerate(np_test):
        test_label = test_labels[i]

        for key, value in sum_dict.items():
            sum_variances = 0
            sum_means = 0
            prob = -2 * log(priori[key])
            for j in range(len(test_vector)):
                sum_variances += log(2 * pi * variance_dict[key][j])
                sum_means += pow(test_vector[j] - means_dict[key][j], 2) / variance_dict[key][j]

            prob += sum_variances + sum_means
            probs[key] = prob

        min_prob = min(probs, key=probs.get)

        if not min_prob == test_label:
            error_count += 1

    recognition_rate = 1 - error_count / len(np_test)
    print(f'Recognition rate is {recognition_rate}')


gaussian()

