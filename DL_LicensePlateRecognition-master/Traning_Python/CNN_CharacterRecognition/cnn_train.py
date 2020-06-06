from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
              'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'P': 24, 'Q': 25, 'R': 26, 'S': 27,
              'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z': 34}

channel = 1
height = 100
width = 75

def cnn_model():
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (height, width, channel)))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Dense(128, activation = 'relu'))
    cnn_model.add(layers.Flatten())
    # 0 - 9 and A-Z => 10 + 25 = 35  -- ignoring O in alphabets.
    cnn_model.add(layers.Dense(35, activation = 'softmax'))
    cnn_model.summary()
    return cnn_model

def get_char_data():
    data = np.array([]).reshape(0, height, width)
    labels = np.array([])
    dirs = [i[0] for i in os.walk('dataset')][1:]
    for dir in dirs:
        image_file_list = glob.glob(dir + '/*.jpg')
        sub_data = np.array([np.array(Image.open(file_name)) for file_name in image_file_list])
        data = np.append(data, sub_data, axis = 0)
        sub_labels = [dictionary[dir[-1:]]] * len(sub_data)
        labels = np.append(labels, sub_labels, axis = 0)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 45, shuffle = True)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_char_data()
x_train = x_train.reshape((x_train.shape[0], height, width, channel))
x_test = x_test.reshape((x_test.shape[0], height, width, channel))
# pixel range is from 0 - 255
x_train, x_test = x_train / 255.0, x_test / 255.0
model = cnn_model()
# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
opt = Adam(lr = 0.001)
model.compile(optimizer = opt, loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
model.save("model_char_recognition.h5")


