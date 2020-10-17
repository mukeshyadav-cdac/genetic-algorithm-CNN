import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from scipy.stats import bernoulli
from bitstring import BitArray
import numpy as np

from deap import base, creator, tools, algorithms
# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

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


def evaulate(individual):
    model = Sequential()
    first_layer_features = BitArray(individual[0:5]).uint
    first_layer_kernel_size = BitArray(individual[5:8]).uint
    second_layer_features = BitArray(individual[8:13]).uint
    second_layer_kernel_size = BitArray(individual[13:]).uint

    if first_layer_kernel_size == 0 or second_layer_kernel_size == 0 or first_layer_features == 0 or second_layer_features == 0:
        return -50,

    model.add(Conv2D(first_layer_features, kernel_size=first_layer_kernel_size,
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(second_layer_features,
                     kernel_size=second_layer_kernel_size, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    his = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=1)
    return his.history['accuracy'][0],


population_size = 8
num_generations = 4
gene_length = 16

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary,
                 n=gene_length)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', evaulate)
hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population = toolbox.population(n=population_size)
r, log = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.1,
                             ngen=num_generations, halloffame=hof, stats=stats)

best_individuals = tools.selBest(population, k=1)

print(hof)
print(log)
print(best_individuals)
