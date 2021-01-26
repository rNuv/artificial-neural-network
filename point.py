#point class to hold inputs
class Point:
    #initialize with x, y and bias points along with label with correct classification
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.b = 1
        self.inputs = [x, y, self.b]
        self.label = label

    #print point 
    def print(self):
        print("(" + str(self.x) + ", " + str(self.y) + ") Answer: " + str(self.label))