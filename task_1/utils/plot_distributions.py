import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(x1, x2, min_v=-300, max_v=800, balance=True, names=['test', 'train']):
    """Plots the distribution between two variables of interest, x1 and x2.

    :param x1: variable 1 (smallest number of samples - if balance = True)
    :type x1: pandas.core.frame.DataFrame 

    :param x2: variable 2  (largest number of samples - if balance = True)
    :type x2: pandas.core.frame.DataFrame

    :param names: list containing strings of dataset names (for plotting) in same order provided
    :type x2: list

    :param balace: to balance the sets based on x1 dimension (dim 0), defaults to True
    :type balace: bool, optional
    """

    if balance:
        try:
            # Check length of x1 <= x2
            assert(x1.shape[0] <= x2.shape[0])

            # Balance data
            n_samples = np.shape(x1)[0]
            x2 = x2.sample(n_samples)

        except ValueError as e:
            print(e)
            print('Make sure x1 has largest number of samples!')
            return None

    # Plot KDE
    fig, ax = plt.subplots()
    sns.kdeplot(x1['gex'].values, label=names[0])
    sns.kdeplot(x2['gex'].values, label=names[1])
    plt.title("GEX Distributions ({}-{})".format(names[0], names[1]))
    plt.xlim([min_v, max_v])
    ax.legend()
    plt.show()


# Example:

# Read in the data
# data_root = 'PATH/TO/task_1/data/CAGE-train'
# files_train = glob.glob(os.path.join(data_root, '*train_y.tsv'))
# files_val = glob.glob(os.path.join(data_root, '*val_y.tsv'))

# # Get train and test
# df_train = pd.concat([pd.read_csv(x, sep='\t') for x in files_train])
# df_val = pd.concat([pd.read_csv(x, sep='\t') for x in files_val])


# Plot distributions
# plot_distribution(x1=df_val, x2=df_train, balance=True,
#                   names=['test', 'train'])
