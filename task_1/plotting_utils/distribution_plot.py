import os
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_loader import *
from plotting_utils.plotting_metrics import euclidean_distance


def plot_kde(x1, x2, min_v=-300, max_v=800, balance=True, names=['test', 'train']):
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

    return fig, ax


def plot_hvg(chromosome=1, n=20, show=True, figsize=(8, 8)):
    """PLots the highly variable (HVG) gex values between cell lines 1 and 2

    :param chromosome: Chromosome of interest, defaults to 1
    :type chromosome: int, optional
    :param n: number of top HVG gex to show, defaults to 20

    :type n: int, optional
    :param show: to show plot or not, defaults to True
    :type show: bool, optional

    :param figsize: set the figure ax for largers n, defaults to (8, 8)
    :type figsize: tuple, optional
    :return: top n highly variable genes between cell lines
    :rtype: pandas.core.DataFrame
    """

    # Get data
    all_genes = load_train_genes()
    train_chr = get_train_chr()
    df = filter_genes_by_chr(all_genes, train_chr)
    assert (int(chromosome) in df.chr.values)

    # Split into cell-lines
    df = df.sort_values(by='gene_name', ascending=True)
    x1 = df[(df.cell_line == 1) & (df.chr == chromosome)]
    x2 = df[(df.cell_line == 2) & (df.chr == chromosome)]

    # Aggregate information
    gene_names = x1['gene_name']
    x = pd.DataFrame(
        {
            'gene_name': x1['gene_name'],
            'cell_1_gex': x1['gex'],
            'cell_2_gex': x2['gex']
        })
    x.set_index('gene_name', inplace=True, drop=True)

    # Get eucliden distances
    maxi = 1
    mini = 0
    dist = np.array(x.apply(lambda x: euclidean_distance(x), axis=1))
    dist_std = (dist - dist.min()) / (dist.max() - dist.min())
    dist_scaled = dist_std * (maxi - mini) + mini

    # Create DataFrame
    dist_df = pd.DataFrame(
        {'gene_name': gene_names,
         'euclidean_dist': dist_scaled})

    # Set index to gene names
    dist_df.set_index('gene_name', inplace=True, drop=True)
    dist_df.sort_values(by=['euclidean_dist'], ascending=False, inplace=True)

    if show:
        ax = dist_df.iloc[:n, :].plot.barh(figsize=figsize)
        ax.invert_yaxis()
        plt.title(
            'Top {} most variable gex between cell-lines for chromosome {}'.format(n, chromosome))
        plt.show()

    return dist_df


def plot_chr_similartiy(show=True):
    """Plots chromosome similarities (euclidean) between cell-lines X1 and X2

    :param show: to show the plot results, defaults to True
    :type show: bool, optional

    :return: chr similarity scores (euclidean)
    :rtype: pandas.core.DataFrame
    """

    # Get data
    all_genes = load_train_genes()
    train_chr = get_train_chr()
    df = filter_genes_by_chr(all_genes, train_chr)

    # Columns to drop
    cols_to_keep = ['gene_name', 'gex', 'chr']
    cols_to_dicard = [x for x in df.columns if x not in cols_to_keep]

    # Split into cell-lines
    df = df.sort_values(by='gene_name', ascending=False)
    x1 = df[(df.cell_line == 1)]
    x2 = df[(df.cell_line == 2)]

    # Remove superfluous columns
    x1 = x1.drop(columns=cols_to_dicard, inplace=False)
    x2 = x2.drop(columns=cols_to_dicard, inplace=False)

    # Compute chr similarity
    chr_sim = []
    for c in train_chr:
        r1 = x1[x1.chr == c].gex.values
        r2 = x2[x2.chr == c].gex.values

        # dont scale on euclidean distances here
        dist = np.linalg.norm([r1, r2])
        chr_sim.append(dist)

    # Format dataframe
    df_sim = pd.DataFrame({'chr': train_chr, 'euclidean_dist': chr_sim})
    df_sim['chr'] = df_sim['chr'].astype('int')
    df_sim = df_sim.sort_values(by='euclidean_dist', ascending=False)
    df_sim.set_index(['chr'], inplace=True, drop=True)

    if show:
        ax = df_sim.plot.barh(figsize=(8, 8))
        ax.invert_yaxis()
        plt.title(
            " Chromosome similarity distance (euclidean) for cell-lines A and B")
        plt.show()

    return df_sim
