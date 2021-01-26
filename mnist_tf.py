import numpy as np
import mnist 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

#load dataset from mnist module
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#normalize data to [0-1]
train_images = train_images / 255
test_images = test_images / 255

#flatten data to 1x784 vector
train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

#create 2 layer model with 784 input nodes, 64 hidden nodes and 10 output nodes
model = keras.Sequential()

#add hidden layer and output layer
model.add( layers.Dense(64, activation='sigmoid', input_dim=784))
model.add( layers.Dense(10, activation='sigmoid'))

#compile the model
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#train the model
model.fit(
    train_images,
    utils.to_categorical(train_labels),
    epochs = 5,
    batch_size = 32
)

#evaluate the model
model.evaluate(
    test_images,
    utils.to_categorical(test_labels),
)