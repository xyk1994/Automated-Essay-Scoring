import numpy as np

def CrossValidation(dataset, n_fold, order):
    whole_size = len(dataset)
    if whole_size % n_fold == 0:
        CVsize = whole_size // n_fold
    else:
        CVsize = (whole_size // n_fold) + 1

    if order == 0:
        test = dataset[0: CVsize]
        train = dataset[CVsize: len(dataset)]
    elif order == n_fold - 1:
        test = dataset[order * CVsize: len(dataset)]
        train = dataset[0: order * CVsize]
    else:
        test = dataset[order * CVsize: (order + 1) * CVsize]
        train = np.concatenate((dataset[0: order * CVsize], dataset[(order + 1) * CVsize: len(dataset)]), axis=0)
    return train, test
