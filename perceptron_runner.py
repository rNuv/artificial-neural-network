import random
import point
import perceptron

#function that we are using to label points (basically the goal of our perceptron to reach)
def function(x):
    return 0.2 * x + 7

#function to print out equation of our line given the weights 
def get_line(weights):
    if(len(weights) >= 3):
        m = (weights[0]/weights[1]) * -1
        b = (weights[2]/weights[1]) * -1
        if b >= 0: 
            return "y=" + str(m) + "x+" + str(b)
        else:
            return "y=" + str(m) + "x" + str(b)
    else:
        return "weights not valid"

#runner
training_size = 7000
points = []

#start with random line and weights
for i in range(0, training_size):
    x = random.randrange(-50, 50)
    y = random.randrange(-100, 100)
    answer = 1 if y >= function(x) else -1
    points.append(point.Point(x, y, answer))

#create perceptron with 3 inputs and 0.1 learning rate
neuron = perceptron.Perceptron(3, 0.1)

#print initial random line
print("initial weights: " + str(neuron.get_weights()))
print("initial line: " + get_line(neuron.get_weights()))

#train each point 100 times
for i in range(100):
    for point in points:
        neuron.train(point.inputs, point.label)

#print trained line (tries to be as close to function as possible)
print("trained weights: " + str(neuron.get_weights()))
print("trained line: " + get_line(neuron.get_weights()))