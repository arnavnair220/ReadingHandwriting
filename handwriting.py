import numpy
import pandas
from matplotlib import pyplot





class NeuralNetwork:

    #Neural Network initialization
    def __init__(self):
        self.first_layer_weights = numpy.random.rand(10, 784)
        self.first_layer_biases = numpy.random.rand(10, 1)
        self.second_layer_weights = numpy.random.rand(10, 10)
        self.second_layer_biases = numpy.random.rand(10, 1)

    def loadData(self, fileLocation):
        #Load in data
        dataset = pandas.read_csv(fileLocation)

        #Convert from dataframe to array
        dataArray = numpy.array(dataset)
        numpy.random.shuffle(dataArray)
        self.numPictures, self.numPixelsPlusLabel = dataArray.shape

        self.labels = dataArray.T[0]
        self.pixelValues = dataArray.T[1:self.numPixelsPlusLabel]
    
    def fixWeightsNBiases(self, learningRate, secondLayerWeightsDifference, secondLayerBiasesDifference, firstLayerWeightsDifference, firstLayerBiasesDifference):
        self.first_layer_weights = self.first_layer_weights - learningRate * firstLayerWeightsDifference
        self.first_layer_biases = self.first_layer_biases - learningRate * firstLayerBiasesDifference
        self.second_layer_weights = self.second_layer_weights - learningRate * secondLayerWeightsDifference
        self.second_layer_biases = self.second_layer_biases - learningRate * secondLayerBiasesDifference

#creates layers
def forwardProp(inputData, neuralNetwork):
    #create first layer
    toBeSigmoided = neuralNetwork.first_layer_weights.dot(inputData)+neuralNetwork.first_layer_biases
    firstLayer = sigmoid_function(toBeSigmoided)
    #create second layer
    toBeSoftmaxed = neuralNetwork.second_layer_weights.dot(inputData)+neuralNetwork.second_layer_biases
    secondLayer = softmax_function(toBeSoftmaxed)

    return toBeSigmoided, firstLayer, toBeSoftmaxed, secondLayer

def backProp(neuralNetwork, toBeSigmoided, firstLayer, toBeSoftmaxed, secondLayer):
    #find error in second layer's weights and biases
    secondLayerDifference = secondLayer - createCorrectAnswerVector(neuralNetwork.labels)
    secondLayerWeightsDifference = (1/neuralNetwork.numPictures) * secondLayerDifference.dot(firstLayer.T)
    secondLayerBiasesDifference = (1/neuralNetwork.numPictures) * numpy.sum(secondLayerDifference, 2)

    #find error in first layer's weights and biases
    firstLayerDifference = neuralNetwork.second_layer_weights.T.dot(secondLayerDifference) * sigmoidDerivative(toBeSigmoided)
    firstLayerWeightsDifference = (1/neuralNetwork.numPictures) * firstLayerDifference.dot(neuralNetwork.pixelValues.T)
    firstLayerBiasesDifference = (1/neuralNetwork.numPictures)* numpy.sum(firstLayerDifference, 2)

    return secondLayerWeightsDifference, secondLayerBiasesDifference, firstLayerWeightsDifference, firstLayerBiasesDifference

def gradient_descent(neuralNetwork, pixelValues, labels, iterations, learningRate):
    for i in range(iterations):
        toBeSigmoided, firstLayer, toBeSoftmaxed, secondLayer = forwardProp(neuralNetwork, )
        

#used on inputted data after weights and biases
def sigmoid_function(inputMatrix):
    return None

#used on first layer output after weights and biases
def softmax_function(inputMatrix):
    return exp(inputMatrix)/(numpy.sum(exp(inputMatrix)))

def createCorrectAnswerVector(labels):
    correctAnswerVector = numpy.zeros((labels.size, labels.max()+1))
    correctAnswerVector[numpy.arrange(labels.size), labels] = 1
    return correctAnswerVector.T

def sigmoidDerivative(inputMatrix):
    return None