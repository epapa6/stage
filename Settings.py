from scikit_weak.feature_selection import DELIN
from scikit_weak.classification import RRLClassifier, WeaklySupervisedKNeighborsClassifier, \
    WeaklySupervisedRadiusClassifier, GRMLinearClassifier


def chose_classifier(
        classifier):
    if classifier == 'DELIN':
        return DELIN(k=5, d=0.75, n_iters=100)
    '''
    if classifier == 'RoughSetSelectorLambda':
        return RoughSetSelector(random_state=0, n_iters=10, search_strategy='approximate',
                                neighborhood='nearest', discrete=False, method='lambda',
                                n_neighbors=5, l=0.5, epsilon=0.05)
    if classifier == 'RoughSetSelectorConservative':
        return RoughSetSelector(random_state=0, n_iters=10, search_strategy='approximate',
                                neighborhood='nearest', discrete=False, method='conservative',
                                n_neighbors=5, epsilon=0.05)
    if classifier == 'RoughSetSelectorDominance':
        return RoughSetSelector(random_state=0, n_iters=10, search_strategy='approximate',
                                neighborhood='nearest', discrete=False, method='dominance',
                                n_neighbors=5, epsilon=0.05)
    
    if classifier == 'GeneticRoughSetSelectorLambda':
        return GeneticRoughSetSelector(random_state=0, discrete=False, n_iters=10, neighborhood='nearest',
                                       n_neighbors=5, method='lambda', l=0.5, epsilon=0.05)
    if classifier == 'GeneticRoughSetSelectorConservative':
        return GeneticRoughSetSelector(random_state=0, discrete=False, n_iters=10, neighborhood='nearest',
                                       n_neighbors=5, method='conservative', epsilon=0.05)
    if classifier == 'GeneticRoughSetSelectorDominance':
        return GeneticRoughSetSelector(random_state=0, discrete=False, n_iters=10, neighborhood='nearest',
                                       n_neighbors=5, method='dominance', epsilon=0.05)
    '''
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
        './Datasets/avila.csv'
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
        './Datasets/hcv.csv',
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
        './Datasets/taiwan.csv',
        './Datasets/thyroid.csv',
        './Datasets/vowel.csv',
        './Datasets/wifi.csv',
        './Datasets/wine.csv'
    ]

    '''
    ERROR:
    './Datasets/20newsgroups.csv',
    './Datasets/micromass.csv'
    '''

    '''
    WARNING:
    './Datasets/cargo.csv',         # The least populated class in y has only 3 members, less than n_split = 5
    './Datasets/qualitywine.csv',   # The least populated class in y has only 2 members, less than n_splits=5
    '''

    return datasets


def max_samples():
    return 1000


def split_cv():
    return 5


def get_iterations():
    return 100
