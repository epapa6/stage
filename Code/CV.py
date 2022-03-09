import logging
import time

import numpy as np
from numpy.core.fromnumeric import mean
from scikit_weak.utils import DiscreteEstimatorSmoother
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

import Settings as st


def main(
        str_model,
        dataset,
        train,
        predict,
        name_dataset,
        k=5):
    model_name = str_model
    split_cv = st.split_cv()
    start_time = time.time()

    acc_array_cv = []
    bal_acc_array_cv = []
    micro_f1_array_cv = []
    macro_f1_array_cv = []

    x, y = dataset

    skf = StratifiedKFold(n_splits=split_cv,
                          shuffle=True,
                          random_state=0)
    np.random.seed(0)
    smt = DiscreteEstimatorSmoother(KNeighborsClassifier(n_neighbors=k),
                                    type='fuzzy')
    y_fuzzy = smt.fit_transform(x, y)

    for train_index, test_index in skf.split(x, y):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        y_fuzzy_train = y_fuzzy[train_index]

        clf = st.chose_classifier(model_name)
        reducer, estimator = train(x_train, y_fuzzy_train, clf)

        result = predict(x_test, y_test, reducer, estimator)
        acc_array_cv.append(result['acc'])
        bal_acc_array_cv.append(result['balacc'])
        micro_f1_array_cv.append(result['microf1'])
        macro_f1_array_cv.append(result['macrof1'])

    execution_time = time.time() - start_time
    acc_cv = mean(acc_array_cv)
    bal_cv = mean(bal_acc_array_cv)
    micro_f1_cv = mean(micro_f1_array_cv)
    macro_f1_cv = mean(macro_f1_array_cv)

    logging.error('{} {} {} {} -------- {}s'.format(name_dataset, str_model,
                                                    np.around(bal_acc_array_cv, decimals=3),
                                                    np.around(bal_cv, decimals=3),
                                                    round(execution_time, 3)))
    return {
        'acc': acc_cv,
        'balacc': bal_cv,
        'microf1': micro_f1_cv,
        'macrof1': macro_f1_cv,
        'time': execution_time
    }
