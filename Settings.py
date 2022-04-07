from scikit_weak.feature_selection import DELIN, RoughSetSelector, GeneticRoughSetSelector


def chose_classifier(
        classifier):
    if classifier == 'DELIN':
        return DELIN(n_iters=10, k=5, d=0.75)

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
        './Datasets/car.csv',
        './Datasets/cancerwisconsin.csv',
        './Datasets/banknote.csv'
    ]

    """
        './Datasets/cargo.csv'
        './Datasets/credit.csv',
        './Datasets/crowd.csv',
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
        './Datasets/obesity.csv',
        './Datasets/occupancy.csv',
        './Datasets/pen.csv',
        './Datasets/qualitywine.csv',
        './Datasets/robot.csv',
        './Datasets/sensorless.csv',
        './Datasets/shill.csv',
        './Datasets/sonar.csv',
        './Datasets/taiwan.csv',
        './Datasets/thyroid.csv',
        './Datasets/vowel.csv',
        './Datasets/wifi.csv',
        './Datasets/wine.csv',
        './Datasets/20newsgroups.csv',
        './Datasets/myocardial.csv',
        './Datasets/micromass.csv'
        """

    return datasets


def max_samples():
    return 1000


def split_cv():
    return 5


def get_iterations():
    return 100
