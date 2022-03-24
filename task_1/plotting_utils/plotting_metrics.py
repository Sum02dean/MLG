import numpy as np


def euclidean_distance(x):
    """Computes euclidean distance between coordinates in x

    :param x: a vecotr of values
    :type x: list or numpy arrau
    :return: scalar value of euclidean distance
    :rtype: scalar
    """
    dist = np.linalg.norm(x)
    return dist
