import os

import pandas as pd
from pandas import DataFrame

from data_loader import load_train_genes_for_cell_line

data_small_folder = "data_small/"


def created_gct_file_train_plus_val() -> dict[int, DataFrame]:
    """
    Create gct files for cell line 1 and cell 2 (train and val data merged) for visualization reason
    IGV only accept gct format expression data(https: // software.broadinstitute.org/software/igv/ExpressionData)

    :return: Dictionary contain gct file for each cell line
    :rtype: pd.DataFrame
    """

    gct_frame_dict = {}
    for cell_line in [1, 2]:
        # notice  here train_genes is combination of train and val data in original dataset
        merged_train_info = load_train_genes_for_cell_line(cell_line=cell_line)
        feature_names = ["gene_name", "chr", "gene_start", "gene_end", "gex"]
        merged_train_list = merged_train_info.loc[:, feature_names].values.tolist()

        gct_list = list()
        for gene_name, chr, gene_start, gene_end, gex, in merged_train_list:
            gct_list.append([gene_name, f" na |@{chr}:{gene_name}-{gene_end}|", gex, ])
        gct_frame_dict[cell_line] = pd.DataFrame(
            gct_list, columns=["name", "Description", "sample 1"])

    return gct_frame_dict


if __name__ == '__main__':

    GCT_frame_dict = created_gct_file_train_plus_val()

    for key in GCT_frame_dict.keys():
        GCT_frame_dict[key].to_csv(os.path.join(data_small_folder, f"X{key}_trainAndVal.gct"), header=True, index=False,
                                   sep="\t")
