import logging
import os

from multiprocessing import Pool

import Settings as st
from Code import CV as cv
from Code import Preprocessing as pr
from Code import Utility as ut

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.ERROR)


def execution(str_model, name_dataset, dataset, train, predict):
    metrics = cv.main(str_model, dataset, train, predict, name_dataset)
    for key in metrics:
        ut.write_on_dict(name_dataset, str_model + " - " + key, metrics[key])


def preparation(url):
    name_dataset = ut.extract_name_dataset(url)
    dataset, n_label, features, row, n_initial_rows = pr.all_preprocesing_steps(url, st.max_samples())

    ut.write_on_dict(name_dataset, 'Dataset', name_dataset)
    ut.write_on_dict(name_dataset, 'Classes', n_label)
    ut.write_on_dict(name_dataset, 'Features', features)
    ut.write_on_dict(name_dataset, 'CurrentRows', row)
    ut.write_on_dict(name_dataset, 'OriginalRows', n_initial_rows)

    logging.warning("{} {} {} {} {}".format(name_dataset, n_label, features, row, n_initial_rows))

    execution('DELIN', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('RRLClassifier', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('WeaklySupervisedKNeighborsClassifier', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('WeaklySupervisedKRadiusClassifier', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('GRMLogistic', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('GRMSVM', name_dataset, dataset, ut.generic_train, ut.normal_predict)
    execution('PseudoLabelsClassifier', name_dataset, dataset, ut.generic_train, ut.normal_predict)


if __name__ == '__main__':
    datasets = st.get_dataset()

    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    if not os.path.exists('./Results/{}'.format('JSON')):
        os.makedirs('./Results/{}'.format('JSON'))

    ut.clean_json(datasets)

    multicore = True
    # multicore = False

    if multicore:
        with Pool(4) as p:
            print(p.map(preparation, datasets))
    else:
        for x in datasets:
            preparation(x)
