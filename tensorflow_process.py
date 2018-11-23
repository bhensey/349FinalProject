import tensorflow as tf
from tensorflow import keras

import random
import numpy as np
import csv

processed_data = open('processed_data.csv','r')

train_data = []
train_labels = []

firstline = processed_data.readline()

myreader = csv.reader(processed_data,delimiter=',')
sublist = {"AskReddit":0,"politics":1,'worldnews':2,
           'nba':3,'funny':4,'movies':5}

for line in myreader:
    train_data.append(np.array(line[2:-1]))
    train_labels.append(np.array(sublist[line[-1]]))

test_data = np.array(train_data[4000:])
test_labels = np.array(train_labels[4000:])
train_data = np.array(train_data[:4000])
train_labels = np.array(train_labels[:4000])

NUM_WORDS = len(train_data[0])

model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works. 
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(6, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

model.summary()

history = model.fit(train_data,
                              train_labels,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_data, test_labels),
                              verbose=1)

results = model.evaluate(test_data,test_labels)

print(results)
