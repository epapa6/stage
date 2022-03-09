import itertools
import json
import os
import re

import numpy as np
from scikit_weak.classification import WeaklySupervisedKNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

np.set_printoptions(suppress=True)


def transpose_array(
        a):
    arr = np.array(a, dtype='object')

    if arr.ndim > 2:
        transposed_array = np.transpose(arr, (1, 0, 2))
    else:
        transposed_array = np.transpose(arr)

    return transposed_array


def generic_train(
        x_train,
        y_train,
        classifier):
    reducer = classifier.fit(x_train, y_train)
    x_red = reducer.transform(x_train)
    estimator = WeaklySupervisedKNeighborsClassifier(k=5)
    estimator.fit(x_red, y_train)

    return reducer, estimator


def normal_predict(
        x_test,
        y_true,
        reducer,
        estimator):
    x_red = reducer.transform(x_test)
    y_pred = estimator.predict(x_red)

    return {
        'acc': accuracy_score(y_true, y_pred),
        'balacc': balanced_accuracy_score(y_true, y_pred),
        'microf1': f1_score(y_true, y_pred, average='micro'),
        'macrof1': f1_score(y_true, y_pred, average='macro')
    }


def grid(
        grid):
    property_value = [grid[x] for x in grid]
    property = [x for x in grid]

    combinations = list(itertools.product(*property_value))
    list_combinations = [dict(zip(property, combination))
                         for combination in combinations]

    return list_combinations


def clean_json(
        datasets):
    dir = './Results/JSON/'

    files_in_dir = os.listdir(dir)
    filtered_files = [
        file for file in files_in_dir if file.endswith(".json")]

    for dataset in datasets:
        file_name = dataset.split("/")[-1]
        file = file_name.split(".")[0] + ".json"
        if file in filtered_files:
            path_to_file = os.path.join(dir, file)
            os.remove(path_to_file)


def write_on_dict(
        key_dataset,
        key,
        value):
    path = "./Results/JSON/{}.json".format(key_dataset)
    if not os.path.exists(path):
        dictionary = {}
        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)

    if os.path.exists(path):
        json_file = open(path, "r")
        dictionary = json.load(json_file)
        json_file.close()

        dictionary[key] = value

        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)


def extract_name_dataset(
        url):
    matches = re.finditer('/', url)
    matches_positions = [match.start() for match in matches]

    last_slash = matches_positions[-1] + 1

    return url[last_slash - 4]
