# Artificial Neural Network

## Description
This is a Python implementation of a very rudimentary 2 layer Neural Network class. The class features a method to feed forward a certain instance for a prediction and another method to train and adjust its weights based on a labelled data instance. I tested the neural network on a linearly inseparable problem like the XOR logic gate. I then tested it on the MNIST database of handwritten digits. With 784 input nodes, 64 hidden nodes and 10 output nodes, the net acheives about 94% accuracy on 10000 test images after being trained with 5 epochs on 60000 training images, which isn't too shabby 🤷 I compared my results with the same neural network implementation in Tensorflow and Tensorflow acheived about an accuracy of about 95-96%. 

## Pictures
![](images/xor.png)
![](images/mnist_nn.png)
![](images/mnist_tf.png)

## Technologies
- ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- MNIST database (http://yann.lecun.com/exdb/mnist/)<br />

---
*Made with <3 by Arnav, circa 2020*
