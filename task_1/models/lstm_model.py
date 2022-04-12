import datetime

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader

from utils.data_loader import load_test_genes, create_submission
from utils.dataset import HistoneDataset
from utils.stratification import chromosome_splits
from utils.tensorflow_utils import spearman_rankcor


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((1, n_histones, n_bins), input_shape=(n_histones, n_bins)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(300, kernel_size=(1, 7), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Reshape((1, 300)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Reshape((1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(lr=init_lr),
        metrics=[spearman_rankcor],
    )
    print(model.summary())
    return model


def get_xy(split_cell_line=None):
    train_genes, valid_genes = chromosome_splits(test_size=0.25)
    if split_cell_line is not None:
        train_genes, valid_genes = chromosome_splits(test_size=0.25, train_cell_line=split_cell_line)

    train_dataloader = DataLoader(
        HistoneDataset(train_genes, left_flank_size=window // 2, right_flank_size=window // 2, bin_size=bin_size),
        shuffle=False, batch_size=len(train_genes))
    valid_dataloader = DataLoader(
        HistoneDataset(valid_genes, left_flank_size=window // 2, right_flank_size=window // 2, bin_size=bin_size),
        shuffle=False, batch_size=len(valid_genes))
    (x_train, y_train) = next(iter(train_dataloader))
    (x_valid, y_valid) = next(iter(valid_dataloader))
    print('train shape', x_train.shape)
    print('valid shape', x_valid.shape)
    return (x_train.numpy(), y_train.numpy()), (x_valid.numpy(), y_valid.numpy())


def get_x_test():
    test_genes = load_test_genes()
    dataloader = DataLoader(
        HistoneDataset(test_genes, left_flank_size=window // 2, right_flank_size=window // 2, bin_size=bin_size),
        shuffle=False, batch_size=len(test_genes))
    x = next(iter(dataloader))
    return test_genes, x.numpy()


def callbacks(filename):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit/' + filename, histogram_freq=1)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='spearman_rankcor', patience=15, mode='max')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoint/{filename}.h5',
        monitor='spearman_rankcor',
        mode='max',
        save_weights_only=False,
        save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

    return [tensorboard_callback, model_checkpoint_callback, early_stop, reduce_lr]


def fit_predict(cell_line):
    model = get_model()
    (x_train, y_train), (x_valid, y_valid) = get_xy(cell_line)

    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_valid, y_valid),
              callbacks=callbacks(time))

    best_model = tf.keras.models.load_model(f'checkpoint/{time}.h5',
                                            custom_objects={"spearman_rankcor": spearman_rankcor})

    train_pred = best_model.predict(x_train)
    valid_pred = best_model.predict(x_valid)
    return float(spearman_rankcor(y_train, train_pred)), float(spearman_rankcor(y_valid, valid_pred))


def predict_test(filename):
    model = tf.keras.models.load_model('checkpoint/' + filename + '.h5',
                                       custom_objects={"spearman_rankcor": spearman_rankcor})

    test_genes, x_test = get_x_test()
    test_pred = model.predict(x_test)

    create_submission(test_genes, test_pred)


def train_all():
    model = get_model()
    (x_train, y_train), (x_valid, y_valid) = get_xy()
    x = np.append(x_train, x_valid, axis=0)
    y = np.append(y_train, y_valid, axis=0)

    filename = 'sub' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # filename = 'sub20220405-230033'
    model.fit(x, y, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks(filename))

    model = tf.keras.models.load_model('checkpoint/' + filename + '.h5',
                                       custom_objects={"spearman_rankcor": spearman_rankcor})
    pred = model.predict(x)
    print(float(spearman_rankcor(y, pred)))
    return filename


if __name__ == '__main__':
    n_histones = 7
    window = 5000
    bin_size = 100
    n_bins = window // bin_size

    n_epochs = 100
    batch_size = 128
    init_lr = 0.005

    filename = 'sub20220405-233457'
    model = tf.keras.models.load_model('checkpoint/' + filename + '.h5',
                                       custom_objects={"spearman_rankcor": spearman_rankcor})
    print(model.summary())

    # model_file = train_all()
    # predict_test('sub20220405-233457')

    # print('mixed cell-line', fit_predict(None))

    # for cell_line in [1, 2]:
    #     print(f'train cell line {cell_line}', fit_predict(cell_line))
