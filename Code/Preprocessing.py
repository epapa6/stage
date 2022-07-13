import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset_to_pandas(path):
    my_data = pd.read_csv(path, header=None)

    logging.info("Dataset shape {}".format(my_data.shape))

    return my_data


def split_xy(my_data):
    n_attribute = my_data.shape[1]
    x = my_data.iloc[:, 0:n_attribute - 1]
    y = my_data.iloc[:, n_attribute - 1:n_attribute]

    x = x.to_numpy()
    y = y.to_numpy().flatten()

    logging.info("X shape {}".format(x.shape))
    logging.info("y shape {}".format(y.shape))
    logging.info("Number of class {}".format(len(set(y))))

    return x, y


def undersampling(my_data, max_samples):
    if my_data.shape[0] > max_samples:
        n_col = my_data.shape[1]
        my_data = my_data.rename(columns={n_col - 1: "target"})

        my_data = my_data.sample(
            n=max_samples, random_state=1).reset_index(drop=True)

    logging.info("Data shape after undersampling {}".format(my_data.shape))

    return my_data


def class_number(y):
    return len(set(y))


def all_preprocessing_steps(path, max_samples=-1, undersample=True):
    my_data = load_dataset_to_pandas(path)
    n_initial_rows = my_data.shape[0]
    if undersample:
        if max_samples == -1:
            max_samples = n_initial_rows
        my_data = undersampling(my_data, max_samples)

    (x, y) = split_xy(my_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(x)

    n_label = class_number(y)
    n_features = x.shape[1]
    n_row = x.shape[0]

    return (scaled, y), n_label, n_features, n_row, n_initial_rows
