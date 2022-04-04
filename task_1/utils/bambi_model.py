import os
import pandas as pd
import numpy as np
import arviz as az
import bambi as bmb
import numpy as np
import scipy

from stratification import chromosome_splits

def to_pandas(x):
    return pd.DataFrame(x.numpy())

def spearman_scoring(y, y_true):
    return scipy.stats.spearmanr(y, y_true)[0]

def create_submission(test_genes: pd.DataFrame, pred: np.array) -> None:
    save_dir = '../data/submissions'
    file_name = 'gex_predicted.csv'  # DO NOT CHANGE THIS
    zip_name = "Kasak_Liine_Project1.zip"
    save_path = f'{save_dir}/{zip_name}'
    compression_options = dict(method="zip", archive_name=file_name)

    test_genes['gex_predicted'] = pred.tolist()
    print(f'Saving submission to path {os.path.abspath(save_dir)}')
    test_genes[['gene_name', 'gex_predicted']].to_csv(save_path, compression=compression_options)


def get_formula(feature_names):
    """Generates the formula required for the bambi generalized linear model (GLM)
    
    :param feature_names: extracted columns names as list of strings
    :type feature_names: list
    
    :return: a string formula containing the GLM functional formulae
    :rtype: string
    """
    template = ['{}'] * (len(feature_names))
    template = " + ".join(template)
    template = template.format(*list(feature_names))
    f = 'y ~ ' + template
    return f

# Import the data from csv
x_train = pd.read_csv('x_train.csv', index_col=0)
y_train = pd.read_csv('y_train.csv', index_col=0)
x_train['y'] = y_train

# # Import test data
# x_test = pd.read_csv('x_test.csv', index_col=0)
# y_test = pd.read_csv('y_test.csv', index_col=0)
# x_test['y'] = y_test
x_test = pd.read_csv('X3_test.csv', index_col=0)
print(x_test)

# x_train.columns = ['col_{}'.format(i) for i in x_train.columns[:-1].values] + ['y']
# x_test.columns = ['col_{}'.format(i) for i in x_test.columns.values]

# # Define parameters
# params = {
#     'family': 'negativebinomial',
#     'chains': 3,
#     'draws': 1000,
#     'tune': 4000}

# # Get the function formula + model
# f = get_formula(x_train.columns[:-1])
# print(f)
# model = bmb.Model(f, x_train, family=params['family'])

# # Fit model
# fitted_model = model.fit(draws=params['draws'], tune=params['tune'],
#                         chains=params['chains'], init='adapt_diag')


# # Combined the data

# # Get mean posterior predictions
# idata = model.predict(fitted_model, data=x_test[:, :-1], inplace=False)
# preds = np.mean(idata.posterior['y_mean'].values, axis=(0, 1)).reshape(-1, 1)
# print(np.shape(preds))
# print(spearman_scoring(preds, y_test))



# # 0.7415649535231114 intra cell-line splits
# # 0.7112918863640766 inter cell-line splits

