import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import neuralnetwork

#load dataset from files
train_images = idx2numpy.convert_from_file("data/training_images")
train_labels = idx2numpy.convert_from_file("data/training_labels")
test_images = idx2numpy.convert_from_file("data/test_images")
test_labels = idx2numpy.convert_from_file("data/test_labels")

#normalize data to [0-1]
train_images = train_images / 255
test_images = test_images / 255

#flatten data to 1x784 vector
train_images = np.reshape(train_images, (-1, 784)) #60000 rows by 784 columns
test_images = np.reshape(test_images, (-1, 784)) #10000 rows by 784 columns

#create neural network with 784 input nodes, 64 hidden nodes, and 10 output nodes, initialize epochs
nn = neuralnetwork.NeuralNetwork(784, 64, 10)
epochs = 5

#train two epochs with training data set
for i in range(epochs):
    for i in range(len(train_images)):
        image = train_images[i].reshape((784, 1))
        target = np.zeros((10, 1))
        target[train_labels[i]] = 1
        nn.train(image, target)
print("done training")

#test neural network with test data and keep count of classified instances
correctly_classified = 0
incorrectly_classified = 0
incorrect_images = []
incorrect_labels = []
for i in range(len(test_images)):
    test_image = test_images[i].reshape((784, 1))
    output = nn.feedforward(test_image)
    prediction = np.where(output == np.amax(output))
    if prediction[0][0] == test_labels[i]:
        correctly_classified = correctly_classified + 1
    else:
        incorrectly_classified = incorrectly_classified + 1
        incorrect_images.append(test_image)
        incorrect_labels.append(prediction[0][0])
print("done testing\n")

#print results
print("correctly classified: " + str(correctly_classified))
print("incorrectly classified: " + str(incorrectly_classified))
print("error rate: " + str(incorrectly_classified/len(test_images)))
print("accuracy: " + str(correctly_classified/len(test_images)))

#code to print out 25 incorrectly classified images
# fig, axes = plt.subplots(5, 5, figsize=(1.5*5,2*5))
# for i in range(25):
#     ax = axes[i//5, i%5]
#     ax.imshow(incorrect_images[i].reshape((28, 28)), cmap='gray')
#     ax.set_title('Incorrect label: {}'.format(incorrect_labels[i]))
# plt.tight_layout()
# plt.show()