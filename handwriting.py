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
        self.first_layer_weights = numpy.random
        self.second_layer_weights = numpy.rando


