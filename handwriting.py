import numpy
import pandas
from matplotlib import pyplot

#Load in data
dataset = pandas.read_csv("mnist_train.csv")

#Convert from dataframe to array
dataArray = numpy.array(dataset)
numpy.random.shuffle(dataArray)

class NeuralNetwork:

    #Neural Network initialization
    def __init__(self):
        self.first_layer_weights = numpy.random.rand(784, 10)
        self.first_layer_biases = numpy.random.rand(10, 1)
        self.second_layer_weights = numpy.random.rand(10, 10)
        self.second_layer_biases = numpy.random.rand(10, 1)

    #used on inputted data after weights and biases
    def sigmoid_function(inputMatrix):
        return None
    
    #used on first layer output after weights and biases
    def softmax_function(inputMatrix):
        return None

    #creates layers
    def forwardProp(self, inputData):
        #create first layer
        toBeSigmoided = self.first_layer_weights.dot(inputData)+self.first_layer_biases
        firstLayer = sigmoid_function(toBeSigmoided)
        #create second layer
        toBeSoftmaxed = self.second_layer_weights.dot(inputData)+self.second_layer_biases
        secondLayer = softmax_function(toBeSoftmaxed)


