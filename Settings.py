from scikit_weak.feature_selection import DELIN
from scikit_weak.classification import RRLClassifier, WeaklySupervisedKNeighborsClassifier, \
    WeaklySupervisedRadiusClassifier, GRMLinearClassifier, PseudoLabelsClassifier


def chose_classifier(
        classifier):
    if classifier == 'DELIN':
        return DELIN(k=5, d=0.75, n_iters=100)

    if classifier == 'RRLClassifier':
        return RRLClassifier(random_state=0, n_estimators=100, resample=True)

    if classifier == 'WeaklySupervisedKNeighborsClassifier':
        return WeaklySupervisedKNeighborsClassifier(k=5)

    if classifier == 'WeaklySupervisedKRadiusClassifier':
        return WeaklySupervisedRadiusClassifier(radius=5)

    if classifier == 'GRMLogistic':
        return GRMLinearClassifier(loss='logistic', max_epochs=10, regularizer='l2', l2=1)

    if classifier == 'GRMSVM':
        return GRMLinearClassifier(loss='hinge', max_epochs=10, regularizer='l2', l2=1)

    if classifier == 'PseudoLabelsClassifier':
        return PseudoLabelsClassifier(n_iterations=10)


def get_grid(
        model_name):
    if model_name == 'DELIN':
        return {
            'k': [3, 5, 7, 10],
            'd': [0.1, 0.25, 0.5, 0.75]
        }
    if model_name == 'RoughSetSelector':
        return {
            'n_neighbors': [3, 5, 7, 10],
            'method': ['conservative', 'lambda', 'dominance'],
            'l': [0.1, 0.25, 0.5, 0.75, 0.9],
            'epsilon': [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]
        }
    if model_name == 'GeneticRoughSetSelector':
        return {
            'n_neighbors': [3, 5, 7, 10],
            'method': ['conservative', 'lambda', 'dominance'],
            'l': [0.1, 0.25, 0.5, 0.75, 0.9],
            'epsilon': [0.0, 0.01, 0.05, 0.1, 0.25, 0.5],
            'n_iters': [100, 1000],
            'p_mutate': [0.01, 0.05, 0.1, 0.25],
            'tournament_size': [0.1, 0.25, 0.5]
        }


def get_dataset():
    datasets = [
        './Datasets/avila.csv',
        './Datasets/banknote.csv',
        './Datasets/cancerwisconsin.csv',
        './Datasets/car.csv',
        './Datasets/credit.csv',
        './Datasets/crowd.csv',
        './Datasets/data0.csv',
        './Datasets/data5.csv',
        './Datasets/data10.csv',
        './Datasets/data25.csv',
        './Datasets/data50.csv',
        './Datasets/diabetes.csv',
        './Datasets/digits.csv',
        './Datasets/frog-family.csv',
        './Datasets/frog-genus.csv',
        './Datasets/frog-species.csv',
        './Datasets/htru.csv',
        './Datasets/ionosfera.csv',
        './Datasets/iranian.csv',
        './Datasets/iris.csv',
        './Datasets/mice.csv',
        './Datasets/mushroom.csv',
        './Datasets/myocardial.csv',
        './Datasets/obesity.csv',
        './Datasets/occupancy.csv',
        './Datasets/pen.csv',
        './Datasets/robot.csv',
        './Datasets/sensorless.csv',
        './Datasets/shill.csv',
        './Datasets/sonar.csv',
        './Datasets/vowel.csv',
        './Datasets/wifi.csv',
        './Datasets/wine.csv'
    ]

    '''
    ERROR:
    './Datasets/20newsgroups.csv',
    './Datasets/micromass.csv',
    
    './Datasets/hcv.csv',           # IndexError: index 3 is out of bounds for axis 1 with size 3
    './Datasets/taiwan.csv',        # ValueError: This solver needs samples of at least 2 classes in the data, but the 
                                    # data contains only one class: 0
    './Datasets/thyroid.csv',       # IndexError: index 2 is out of bounds for axis 1 with size 2
    '''

    '''
    WARNING:
    './Datasets/cargo.csv',         # The least populated class in y has only 3 members, less than n_split = 5
    './Datasets/qualitywine.csv',   # The least populated class in y has only 2 members, less than n_splits = 5
    '''

    return datasets


def max_samples():
    return 1000


def split_cv():
    return 5


def get_iterations():
    return 100
