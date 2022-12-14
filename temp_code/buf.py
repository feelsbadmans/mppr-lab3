import time
from typing import *
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
import keras
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Dataset:
# https://www.nist.gov/node/1298471/mnist-dataset
# https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip

mnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def cnn_print_digit(d):
    print(d.shape)
    for x in range(28):
        s = ""
        for y in range(28):
            s += "{0:.1f} ".format(d[28*y + x])
        print(s)


def cnn_print_digit_2d(d):
    print(d.shape)
    for y in range(d.shape[0]):
        s = ""
        for x in range(d.shape[1]):
            s += "{0:.1f} ".format(d[x][y])
        print(s)


def mnist_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def mnist_train(model):
    t_start = time.time()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train_OH = np_utils.to_categorical(y_train)
    y_test_OH = np_utils.to_categorical(y_test)
    y_train_OH.shape, y_test_OH.shape

    X_train = X_train / 256
    X_test = X_test / 256

    # Test:

    X_train, X_validate, y_label, y_validate = train_test_split(
        X_train, y_train_OH, test_size=0.5, random_state=2)
    
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(X_train, y_label,
              epochs=40, batch_size=64,
              callbacks=[learning_rate_reduction],
              validation_data=(X_validate, y_validate))

    print("Training done, dT:", time.time() - t_start)


def mnist_predict(model, image_file):
    img = keras.preprocessing.image.load_img(
        image_file, target_size=(28, 28), color_mode='grayscale')
    mnist_predict_img(model, img)


def mnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(mnist_labels[result[0]])


def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(
        img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * \
                np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(
                letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    
    return letters


def img_to_str(model: Any, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - \
            letters[i][1] if i < len(letters) - 1 else 0
        s_out += mnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out


if __name__ == "__main__":
    # model = mnist_model()
    # mnist_train(model)
    # model.save('mnist_letters2.h5')

    model = keras.models.load_model('mnist_letters2.h5')
    s_out = img_to_str(model, "ts.png")
    print(s_out)
