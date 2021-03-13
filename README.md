# Artificial-Neural-Network
A Simple 2 layered Neural Network library made from scratch in Python.

## 😄 Summary 
This is a Python implementation of a very rudimentary 2 layer Neural Network class. The class features a method to feed forward a certain instance for a prediction and another method to train and adjust its weights based on a labelled data instance. I tested the neural network on a linearly inseparable problem like the XOR logic gate. I then tested it on the MNIST database of handwritten digits. With 784 input nodes, 64 hidden nodes and 10 output nodes, the net acheives about 94% accuracy on 10000 test images after being trained with 5 epochs on 60000 training images, which isn't too shabby 🤷 I compared my results with the same neural network implementation in Tensorflow and Tensorflow acheived about an accuracy of about 95-96%. 

## 💻 Tech 
-NumPy for matrix math <br />
-MNIST database (http://yann.lecun.com/exdb/mnist/)<br />
-Tensorflow

## 📷 Pictures 
![](images/xor.png)
![](images/mnist_nn.png)
![](images/mnist_tf.png)
