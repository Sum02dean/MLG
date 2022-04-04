from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import numpy as np
import pandas as pd
import time
import os


def create_submission(test_genes: pd.DataFrame, pred: np.array) -> None:
    save_dir = '../data/submissions'
    file_name = 'gex_predicted.csv'  # DO NOT CHANGE THIS
    zip_name = "Tao_Fang_Project1.zip"
    save_path = f'{save_dir}/{zip_name}'
    compression_options = dict(method="zip", archive_name=file_name)

    test_genes['gex_predicted'] = pred.tolist()
    print(f'Saving submission to path {os.path.abspath(save_dir)}')
    test_genes[['gene_name', 'gex_predicted']].to_csv(
        save_path, compression=compression_options)


def get_reshaped_data(dataloader, is_train=True):
    """Reshape data to fit into DeepHistone Model 

    :param dataloader: HistoneDataset
    :type dataloader: HistoneDataset wraped by torch.utils.data.DataLoader
    """
    if is_train:
        # this step is slow since it actually loads all data
        (x, y, genename) = next(iter(dataloader))
    else:
        (x, genename) = next(iter(dataloader))
    x_histone, x_seq = x
    # print(x_histone.shape,x_seq.shape,y.shape,len(genename),len(set(genename)))

    n_genes, n_features_histone, n_bins_histone = x_histone.shape
    x_histone = x_histone.reshape(
        n_genes, 1, n_features_histone, n_bins_histone)

    _, n_bins_seq, n_features_seq = x_seq.shape
    x_seq = x_seq.reshape(n_genes, 1, n_features_seq, n_bins_seq)

    if is_train:
        y = y.reshape(n_genes, 1, 1)
        return(x_histone, x_seq, y, list(genename))
    else:
        return(x_histone, x_seq, list(genename))


def get_dict_from_data(train_index, valid_index, test_index, train, valid, test):
    """_summary_

    :param train_index: _description_
    :type train_index: _type_
    :param valid_index: _description_
    :type valid_index: _type_
    :param test_index: _description_
    :type test_index: _type_
    :param train: _description_
    :type train: _type_
    :param valid: _description_
    :type valid: _type_
    :param test: _description_
    :type test: _type_
    """
    return_dict = {train_index[i]: train[i, ...]
                   for i in range(train.shape[0])}
    # print(len(return_dict))

    return_dict.update({valid_index[i]: valid[i, ...]
                       for i in range(valid.shape[0])})
    # print(len(return_dict))

    return_dict.update({test_index[i]: test[i, ...]
                       for i in range(test.shape[0])})
    # print(len(return_dict))

    return(return_dict)


def get_dict_from_data_submisson(submission_index, submission):
    return_dict = {submission_index[i]: submission[i, ...]
                   for i in range(submission.shape[0])}
    return(return_dict)


def save_model(model, epoch, model_save_folder="../data/DeepHistone/", prefix="", suffix=""):
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    model.save_model(
        f"{model_save_folder}{prefix}time{time_stamp}-epoch{epoch}-model-{suffix}.txt")


def get_compplex_prefix(**kwargs):
    complex_prefix = ""
    for k, v in kwargs.items():
        complex_prefix += f"{k}-{v}-"

    return(complex_prefix)


def loadRegions(regions_indexs, dna_dict, dns_dict, label_dict=None,):
    if dna_dict is not None:
        dna_regions = np.concatenate([dna_dict[meta]
                                     for meta in regions_indexs], axis=0)
    else:
        dna_regions = []
    if dns_dict is not None:
        dns_regions = np.concatenate([dns_dict[meta]
                                     for meta in regions_indexs], axis=0)
    else:
        dns_regions = []

    if label_dict is not None:
        # .astype(int) ; here our output is regression value
        label_regions = np.concatenate(
            [label_dict[meta] for meta in regions_indexs], axis=0)
        return dna_regions, dns_regions, label_regions
    else:
        return dna_regions, dns_regions


def model_train(regions, model, batchsize, dna_dict, dns_dict, label_dict,):
    train_loss = []
    regions_len = len(regions)
    for i in range(0, regions_len, batchsize):
        # for testing reason add this
        if i % 100 == 0:  # 100, 1000,5000
            print(f"batch_idx: {i}")
        regions_batch = [regions[i+j]
                         for j in range(batchsize) if (i+j) < regions_len]
        #print("region_batch: ",(regions_batch))
        seq_batch, dns_batch, lab_batch = loadRegions(
            regions_batch, dna_dict, dns_dict, label_dict)
        _loss = model.train_on_batch(seq_batch, dns_batch, lab_batch)
        train_loss.append(_loss)
    return np.mean(train_loss)


def model_eval(regions, model, batchsize, dna_dict, dns_dict, label_dict,):
    loss = []
    pred = []
    lab = []
    regions_len = len(regions)
    for i in range(0, regions_len, batchsize):
        regions_batch = [regions[i+j]
                         for j in range(batchsize) if (i+j) < regions_len]
        seq_batch, dns_batch, lab_batch = loadRegions(
            regions_batch, dna_dict, dns_dict, label_dict)
        _loss, _pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
        loss.append(_loss)
        lab.extend(lab_batch)
        pred.extend(_pred)
    return np.mean(loss), np.array(lab), np.array(pred)


def model_predict(regions, model, batchsize, dna_dict, dns_dict, label_dict=None,):
    lab = []
    pred = []
    regions_len = len(regions)
    for i in range(0, len(regions), batchsize):
        regions_batch = [regions[i+j]
                         for j in range(batchsize) if (i+j) < regions_len]
        if label_dict is not None:
            seq_batch, dns_batch, lab_batch = loadRegions(
                regions_batch, dna_dict, dns_dict, label_dict)
            _pred = model.test_on_batch(seq_batch, dns_batch)
            lab.extend(lab_batch)
            pred.extend(_pred)
        else:
            seq_batch, dns_batch = loadRegions(
                regions_batch, dna_dict, dns_dict, label_dict)
            _pred = model.test_on_batch(seq_batch, dns_batch)
            pred.extend(_pred)
    if label_dict is not None:
        return np.array(lab), np.array(pred)
    else:
        return np.array(pred)
