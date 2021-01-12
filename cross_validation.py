import numpy as np
import os
from scipy.spatial.distance import sqeuclidean
import matplotlib.pyplot as plt

SEED = 45


def shuffle_data(data: dict) -> dict:
    """Returns uniformly random permutation of <data>."""

    np.random.seed(SEED)  # set seed to get reproducible result
    shuffler = np.random.permutation(len(data['X']))
    return {'X': data['X'][shuffler], 't': data['t'][shuffler]}


def split_data(data: dict, num_folds: int, fold: int) -> tuple:
    """Splits <data> into <num_folds> blocks.
       Returns the selected <fold>_th block as data_fold, and the remaining data
       as data_rest."""

    # split <data> into <num_folds> blocks
    data_splited = {'X': np.array_split(
        data['X'], num_folds), 't': np.array_split(data['t'], num_folds)}

    # store <fold>_th block in data_fold
    data_fold = {'X': data_splited['X'][fold-1],
                 't': data_splited['t'][fold-1]}
    # concatenate and store remaining data
    X = np.delete(data_splited['X'], fold-1, 0)
    t = np.delete(data_splited['t'], fold-1, 0)
    data_rest = {'X': np.concatenate(
        X, axis=0), 't': np.concatenate(t, axis=0)}

    return data_fold, data_rest


def train_model(data: dict, lambd: float) -> np.array:
    """Returns the coefficients of ridge regression with input <data> 
    and penalty level <lambd>."""

    X, t = data['X'], data['t']
    # matrix = X^T X + lambd * N I
    matrix = X.T @ X + lambd * len(X) * np.identity(len(X[0]))
    # model = matrix^(-1) X^T t
    model = np.linalg.inv(matrix) @ X.T @ t
    return model


def predict(data: dict, model: np.array) -> np.array:
    """Returns the predictions based on <data> and <model>."""

    return data['X'] @ model


def loss(data: dict, model: np.array) -> float:
    """Returns the average squared error loss based on <data> and <model>"""

    # squared Euclidean distance between prediction and target
    squared_d = sqeuclidean(predict(data, model), data['t'])
    # average squared error loss
    error = squared_d / (2*len(data['X']))
    return error


def cross_validation(data: dict, num_folds: int, lambd_seq: np.array) -> list:
    """Returns the <num_folds> fold cross validation error based on input 
    <data>, cross all lambd in <lambda_seq>."""

    data = shuffle_data(data)
    cv_error = []
    for lambd in lambd_seq:
        cv_loss_lmd = 0

        # compute CV error for this lambd
        for fold in range(1, num_folds+1):

            # split validation and training data
            val_cv, train_cv = split_data(data, num_folds, fold)
            # train model and compute the validation error
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)

        # average validation errors to get CV error
        cv_error.append(cv_loss_lmd / num_folds)

    return cv_error


if __name__ == "__main__":
    # question 4(a) load data
    file_path = 'data'
    data_train = {'X': np.genfromtxt(os.path.join(file_path, 'data_train_X.csv'), delimiter=','),
                  't': np.genfromtxt(os.path.join(file_path, 'data_train_y.csv'), delimiter=',')}
    data_test = {'X': np.genfromtxt(os.path.join(file_path, 'data_test_X.csv'), delimiter=','),
                 't': np.genfromtxt(os.path.join(file_path, 'data_test_y.csv'), delimiter=',')}

    lambd_seq = np.linspace(0.00005, 0.005, num=50)

    # question 4(c) training and test errors for each lambd
    train_error, test_error = [], []
    for lambd in lambd_seq:
        model = train_model(data_train, lambd)
        train_error.append(loss(data_train, model))
        test_error.append(loss(data_test, model))

    # report training and test errors for each lambd
    fig, ax = plt.subplots()
    ax.plot(lambd_seq, train_error, '-ro')
    ax.plot(lambd_seq, test_error, '-go')
    ax.xaxis.set_label_text('lambd')
    ax.yaxis.set_label_text('errors')
    ax.set_title('Errors for each lambda')
    ax.legend(['train_error', 'test_error'])
    plt.xticks(lambd_seq, fontsize='xx-small', rotation=60)
    plt.savefig('./4c. Train and test errors for each lambda.png')

    # question 4(d) 5_fold and 10_fold CV errors for each lambd
    cv5_error = cross_validation(data_train, 5, lambd_seq)
    lambd_cv5 = lambd_seq[cv5_error.index(min(cv5_error))]
    print('The proposed lambda by 5-fold cross validation is {}'.format(lambd_cv5))

    cv10_error = cross_validation(data_train, 10, lambd_seq)
    lambd_cv10 = lambd_seq[cv10_error.index(min(cv10_error))]
    print('The proposed lambda by 10-fold cross validation is {}'.format(lambd_cv10))

    # plot training, test, 5_fold and 10_fold CV errors for each lambd
    fig, ax = plt.subplots()

    ax.plot(lambd_seq, train_error, '-r')
    ax.plot(lambd_seq, test_error, '-g')
    ax.plot(lambd_seq, cv5_error, '-b')
    ax.plot(lambd_seq, cv10_error, '-k')

    ax.xaxis.set_label_text('lambd')
    ax.yaxis.set_label_text('errors')
    ax.set_title('Errors for each lambda')
    ax.legend(['train_error', 'test_error',
               '5_fold cross validation error', '10_fold cross validation error'])
    plt.savefig('./4d. Errors for each lambda.png')
