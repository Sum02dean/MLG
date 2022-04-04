import datetime

import tensorflow as tf
from keras import backend as K
from torch.utils.data import DataLoader

from utils.dataset import HistoneDataset
from utils.stratification import chromosome_splits
from utils.tensorflow_utils import spearman_rankcor


def get_model(n_histones, n_bins):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((1, n_histones, n_bins), input_shape=(n_histones, n_bins)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(252, kernel_size=(1, 7), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Reshape((1, 252)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Reshape((1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # tf.keras.layers.Attention(),
        # tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(lr=0.005),
        metrics=['accuracy', spearman_rankcor],
    )
    print(model.summary())
    return model


def get_xy(split_cell_line=None):
    train_genes, valid_genes = chromosome_splits(test_size=0.25)
    if split_cell_line is not None:
        train_genes, valid_genes = chromosome_splits(test_size=0.25, train_cell_line=split_cell_line)

    train_dataloader = DataLoader(
        HistoneDataset(train_genes, left_flank_size=5000, right_flank_size=5000, bin_size=100), shuffle=False,
        batch_size=len(train_genes))
    valid_dataloader = DataLoader(
        HistoneDataset(valid_genes, left_flank_size=5000, right_flank_size=5000, bin_size=100), shuffle=False,
        batch_size=len(valid_genes))
    (x_train, y_train) = next(iter(train_dataloader))
    (x_valid, y_valid) = next(iter(valid_dataloader))
    print('train shape', x_train.shape)
    print('valid shape', x_valid.shape)
    return (x_train.numpy(), y_train.numpy()), (x_valid.numpy(), y_valid.numpy())


if __name__ == '__main__':
    # interesting runs:
    # 20220404-033805 only conv
    # 20220404-035714 lstm 32
    # 20220404-051614 252 conv, 32 lstm, batch corr before conv and lstm, end dense relu
    # 20220404-052451 separate by cell line
    # 20220404-053027 same but lstm 64 followed by 32, increased dropout - 0.77 brought in diff cell line!

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_epochs = 500
    batch_size = 128

    (x_train, y_train), (x_valid, y_valid) = get_xy(1)
    model = get_model(7, 100)

    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/fit/' + time, histogram_freq=1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoint/{time}.h5',
        monitor='val_spearman_rankcor',
        mode='max',
        save_best_only=True)

    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_valid, y_valid),
              callbacks=[tensorboard_callback, model_checkpoint_callback])
