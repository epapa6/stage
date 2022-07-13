import itertools
import json
import os
import re

import numpy as np
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
    classifier.fit(x_train, y_train)

    return classifier


def normal_predict(
        x_test,
        y_true,
        classifier):
    y_pred = classifier.predict(x_test)

    return {
        'acc': accuracy_score(y_true, y_pred),
        'balacc': balanced_accuracy_score(y_true, y_pred),
        'microf1': f1_score(y_true, y_pred, average='micro'),
        'macrof1': f1_score(y_true, y_pred, average='macro')
    }


def grid(
        gr):
    prop_value = [gr[x] for x in gr]
    prop = [x for x in gr]

    combinations = list(itertools.product(*prop_value))
    list_combinations = [dict(zip(prop, combination))
                         for combination in combinations]

    return list_combinations


def clean_json(
        datasets):
    directory = './Results/JSON/'

    files_in_dir = os.listdir(directory)
    filtered_files = [
        file for file in files_in_dir if file.endswith(".json")]

    for dataset in datasets:
        file_name = dataset.split("/")[-1]
        file = file_name.split(".")[0] + ".json"
        if file in filtered_files:
            path_to_file = os.path.join(directory, file)
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


def extract_name_dataset(url):
    matches = re.finditer('/', url)
    matches_positions = [match.start() for match in matches]

    last_slash = matches_positions[-1] + 1

    return url[last_slash:-4]
