import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# read train data
train_data = pd.read_csv("./train.csv", header=0, index_col=0, sep=",")
train_X = np.array(train_data.iloc[:, 2:])
train_Y = np.array(train_data.iloc[:, 1])
print(train_X.shape, train_Y.shape)

train_data.head(n=3)

# read test data
test_data = pd.read_csv("./test.csv", header=0, index_col=0, sep=",")
test_X = np.array(test_data.iloc[:, 1:])
print(test_X.shape)
test_data.head(n=3)


def create_submission(df):
    filename = 'submission'
    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    df.to_csv(f'{filename}.zip', compression=compression_options, index=True)


def fit_predict(model, parameters):
    # fit model
    clr = GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=1)
    clr.fit(train_X, train_Y)
    print(clr)
    print(clr.best_params_)

    # check performance on training data
    train_y_pred = clr.predict(train_X)
    train_rmse = mean_squared_error(train_Y, train_y_pred) ** 0.5
    print(train_rmse)

    # write test set predictions to csv
    test_prediction = pd.DataFrame({'Id': test_data['Id'], 'y': clr.predict(test_X)})
    create_submission(test_prediction)


def linear_regression():
    lr_model = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    parameters = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }
    fit_predict(lr_model, parameters)


def ridge_regression():
    ridge_model = linear_model.Ridge(max_iter=15000)
    parameters = {
        'alpha': [0, 0.5, 1],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'solver': ['cholesky', 'svd', 'lsqr']
    }

    fit_predict(ridge_model, parameters)


# run to fit and generate submission file
ridge_regression()
