import random

#perceptron class 
class Perceptron:
    #initialize size of dataset and learning rate
    def __init__(self, inputsize, learning_rate):
      self.weights = [random.uniform(-1.0, 1.0) for i in range(0, inputsize)]
      self.learning_rate = learning_rate

    #method that returns activated sum of linear equation
    def feedforward(self, inputs):
        sum = 0
        for i in range(0, len(inputs)):
            sum += self.weights[i] * inputs[i]
        return self.activate(sum)

    #train weights given a point
    def train(self, inputs, desired):
        #input given point into weighted equation
        guess = self.feedforward(inputs) 
        #calculate error based on actual point 
        error = desired - guess
        #loop through weights and adjust with learning rate slowing down 
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.learning_rate
   
    #activation function (step function)
    def activate(self, n):
        if n >= 0: 
            return 1 
        else: 
            return -1
    
    #return weights
    def get_weights(self):
        return self.weights