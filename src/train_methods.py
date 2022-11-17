import time
from src.constants import MNIST_LABELS

from tensorflow import keras
import idx2numpy
import numpy as np


def mnist_train(model):
    t_start = time.time()

    mnist_path = 'mnist/'
    X_train = idx2numpy.convert_from_file(
        mnist_path + 'emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(
        mnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

    X_test = idx2numpy.convert_from_file(
        mnist_path + 'emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(
        mnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape,
          y_test.shape, len(MNIST_LABELS))

    k = 10
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(MNIST_LABELS))
    y_test_cat = keras.utils.to_categorical(y_test, len(MNIST_LABELS))

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat),
              callbacks=[learning_rate_reduction], batch_size=32, epochs=30)
    print("Training done, dT:", time.time() - t_start)