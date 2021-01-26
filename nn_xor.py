import numpy as np
import random
import neuralnetwork

training_data = [
    {
        "inputs": np.array([[1], [0]]),
        "target": np.array([1])
    },
    {
        "inputs": np.array([[0], [1]]),
        "target": np.array([1])
    },
    {
        "inputs": np.array([[1], [1]]),
        "target": np.array([0])
    },
    {
        "inputs": np.array([[0], [0]]),
        "target": np.array([0])
    }
]

brain = neuralnetwork.NeuralNetwork(2, 2, 1)

for i in range(50000):
    train = random.choice(training_data)
    brain.train(train["inputs"], train["target"])

print(brain.feedforward(np.array([[1], [0]])))
print(brain.feedforward(np.array([[0], [1]])))
print(brain.feedforward(np.array([[1], [1]])))
print(brain.feedforward(np.array([[0], [0]])))
