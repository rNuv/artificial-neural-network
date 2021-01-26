import numpy as np

#neural network class with three layers
class NeuralNetwork:
    #initialize with number of nodes in each of the three layers
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        #number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.3

        #initialize weight matrices with random values
        self.weights_ih = -2 * np.random.rand(self.hidden_nodes, self.input_nodes) + 1
        self.weights_ho = -2 * np.random.rand(self.output_nodes, self.hidden_nodes) + 1

        #initialize bias matrices with random values
        self.bias_h = -2 * np.random.rand(self.hidden_nodes, 1) + 1
        self.bias_o = -2 * np.random.rand(self.output_nodes, 1) + 1
    
    def feedforward(self, input_value):
        #matrix multiply weights with inputs
        hidden_weighted_sum = np.matmul(self.weights_ih, input_value)
        #add bias to weighted sum matrix
        hidden_weighted_sum = np.add(hidden_weighted_sum, self.bias_h)
        #activate the output of hidden layer
        hidden_output = self.sigmoid(hidden_weighted_sum)
        
        #matrix multiply weights with hidden outputs
        output_weighted_sum = np.matmul(self.weights_ho, hidden_output)
        #add bias to weighted sum matrix
        output_weighted_sum = np.add(output_weighted_sum, self.bias_o)
        #activate the output of output layer
        output = self.sigmoid(output_weighted_sum)

        return output

    def train(self, inputs, answer):
        #get initial guess with feed forward methodology
        hidden_weighted_sum = np.matmul(self.weights_ih, inputs)
        hidden_weighted_sum = np.add(hidden_weighted_sum, self.bias_h)
        hidden_output = self.sigmoid(hidden_weighted_sum)
        
        output_weighted_sum = np.matmul(self.weights_ho, hidden_output)
        output_weighted_sum = np.add(output_weighted_sum, self.bias_o)
        output = self.sigmoid(output_weighted_sum)
        
        #calculate the error of initial guess
        output_error = np.subtract(answer, output)
        #calculate hidden errors with dot product of transpose of weight_ho and output_error
        hidden_error = np.matmul(np.transpose(self.weights_ho), output_error)

        #calculate gradient 
        gradient = self.dsigmoid(output)
        #calculate delta bias_o
        delta_bo = self.learning_rate * output_error * gradient
        #add delta_bo to bias_o
        self.bias_o = np.add(self.bias_o, delta_bo)
        #calculate delta weights for hidden-output by multiplying gradient, error, and learning rate with hidden output        
        delta_who = np.matmul(delta_bo, np.transpose(hidden_output))
        #add deltas to weights
        self.weights_ho = np.add(self.weights_ho, delta_who)

        #calculate hidden gradient
        hidden_gradient = self.dsigmoid(hidden_output)
        #calculate delta bias_h
        delta_bh = self.learning_rate * hidden_error * hidden_gradient
        #add delta_bh to bias_h
        self.bias_h = np.add(self.bias_h, delta_bh)
        #calculate delta weights for input by multiplying gradient, hidden error, and learning rate with input
        delta_wih = np.matmul(delta_bh, np.transpose(inputs))
        #add deltas to weights
        self.weights_ih = np.add(self.weights_ih, delta_wih)
        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, y):
        return y * (1 - y)