import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(x1, x2, min_v=-300, max_v=800, balance=True, names=['test', 'train']):
    """Plots the distribution between two variables of interest, x1 and x2.

    :param x1: variable 1 (smallest number of samples - if balance = True)
    :type x1: pandas.core.frame.DataFrame 

    :param x2: variable 2  (lagestest number of samples - if balance = True)
    :type x2: pandas.core.frame.DataFrame

    :param names: list containing strings of dataset names (for plotting) in same order provided
    :type names: list

    :param balace: to balance the sets based on x1 dimension (dim 0), defaults to True
    :type balace: bool, optional
    """

    if balance:
        # Balance data
        n_x1 = np.shape(x1)[0]
        n_x2 = np.shape(x2)[0]

        # Balance by smallest set
        if n_x1 <= n_x2:
            x2 = x2.sample(n_x1)
        else:
            x1 = x1.sample(n_x2)

    # Plot KDE
    fig, ax = plt.subplots()
    sns.kdeplot(x1.values, label=names[0])
    sns.kdeplot(x2.values, label=names[1])
    plt.title("Data Distributions ({}-{})".format(names[0], names[1]))
    plt.xlim([min_v, max_v])
    ax.legend()
    plt.show()
