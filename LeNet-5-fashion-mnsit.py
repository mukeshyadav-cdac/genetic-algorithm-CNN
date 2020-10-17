

from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from scipy.stats import bernoulli
from bitstring import BitArray
import numpy as np
fashion_mnist = tf.keras.datasets.fashion_mnist

# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train[0:10000]
y_train = y_train[0:10000]

# reshape data to fit model
X_train = X_train.reshape(10000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
his = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=1)
print(his.history['accuracy'][0])
