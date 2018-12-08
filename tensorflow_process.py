import tensorflow as tf
from tensorflow import keras

import random
import numpy as np
import csv

processed_data = open('processed_data.csv','r')

data_X = []
data_labels = []

firstline = processed_data.readline()

myreader = csv.reader(processed_data,delimiter=',')
TFMap = {'True':1,'False':0}
sublist = {"AskReddit":0,"politics":1,'worldnews':2,
           'nba':3,'funny':4,'movies':5}

for line in myreader:
    data_X.append(np.array([int(x) for x in line[:-1]]))
    data_labels.append(np.array(sublist[line[-1]]))

#test_data = np.array(train_data[4000:])
#test_labels = np.array(train_labels[4000:])
#train_data = np.array(train_data[:4000])
#train_labels = np.array(train_labels[:4000])

NUM_WORDS = len(data_X[0])

test_size = len(data_X)//10
print("beginning cross validation:")
scores = [];
loss = []
for i in range(10):

    test_data = np.array(data_X[i*test_size:((i+1)*test_size+1)])
    test_labels = np.array(data_labels[i*test_size:((i+1)*test_size+1)])
    train_data = np.array(data_X[:i*test_size] + data_X[(i+1)*test_size+1:])
    train_labels = np.array(data_labels[:i*test_size] +
                            data_labels[(i+1)*test_size+1:])
    model = keras.Sequential([
        # `input_shape` is only required here so that `.summary` works. 
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        #keras.layers.Dropout(0.3),
        #keras.layers.Dense(16, activation=tf.nn.relu),
        #keras.layers.Dropout(0.3),
        keras.layers.Dense(6, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    model.summary()
    print("train size: ",len(train_data),"test size: ", len(test_data))
    history = model.fit(train_data,
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=1)

    results = model.evaluate(test_data,test_labels)
    scores.append(results[1])
    loss.append(results[0])
    print("results",results)
print("mean score",np.mean(scores),"stdev",np.std(scores))
print("mean loss:", np.mean(loss))

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

history_dict = history.history
history_dict.keys()

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

