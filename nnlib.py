import random
import math
import os
import subprocess

# Importing numpy

try:

    import numpy

except ModuleNotFoundError:

    try:

        answer = subprocess.check_output('pip3 install numpy', shell=True).read()
        
    except subprocess.CalledProcessError:

        os.system('sudo apt install python3-pip')
        os.system('pip3 install numpy')

    import numpy

class NeuralNetwork:

    # You need to organize the layers quantity and the num of neurons of each layer

    def __init__(self):

        # That is us default values

        self.returnZeroOrOne = False
        self.learningRate = 1
        self.numOfLayers = 3
        self.numOfNeurons = [784, 16, 10]
        self.programName = 'neuralnetwork'

    def generateNeurons(self):

        self.neurons = []

        for i in range(self.numOfLayers):
            self.neurons.append([])
            for j in range(self.numOfNeurons[i]):
                self.neurons[i].append(0)

    # Now we have to compute the initials weights.

    def generateRandWeights(self):

        self.weights = []

        for i in range(self.numOfLayers - 1):
            self.weights.append([])
            for j in range(self.numOfNeurons[i]):
                self.weights[i].append([])
                for k in range(self.numOfNeurons[i+1]):
                    self.weights[i][j].append(random.random()*2-1) # Generates a value between -1 and 1.
    
    # Now we have to compute the initials bias.

    def storeWeights(self):

        numpyWeights = numpy.array(self.weights)
        numpy.save(self.programName + '_weights', numpyWeights)


    def useStoredWeights(self):

        try:
            self.weights = numpy.load(self.programName + '_weights.npy')

        except FileNotFoundError:
            
            self.generateRandWeights()
            self.storeWeights()


    def storeBias(self):

        numpyBias = numpy.array(self.bias)
        numpy.save(self.programName + '_bias', numpyBias)


    def useStoredBias(self):

        try:
            self.bias = numpy.load(self.programName + '_bias.npy')

        except FileNotFoundError:
            
            self.generateRandBias()
            self.storeBias()

    def generateRandBias(self):

        self.bias = []

        for i in range(self.numOfLayers):
            self.bias.append([])
            for j in range(self.numOfNeurons[i]):
                if i == 0:
                    self.bias[i].append(0)
                else:
                    self.bias[i].append(random.random()*2-1) # Generates a value between -1 and 1.

    def sigmoid(self, value):
        if self.returnZeroOrOne == True:
            if value >= 0:
                return 1
            else:
                return 0
        else:
            try:
                return 1/(1+math.exp(-value))
            except OverflowError:
                return 1/1+math.exp(709)

    def propagate(self, layerToPropagate):

        for i in range(self.numOfNeurons[layerToPropagate+1]):
            newValue = 0
            for j in range(self.numOfNeurons[layerToPropagate]):
                newValue += self.weights[layerToPropagate][j][i] * self.neurons[layerToPropagate][j]
            newValue += self.bias[layerToPropagate][i]
            newValue = self.sigmoid(newValue)
            self.neurons[layerToPropagate+1][i] = newValue

    # Getting a guess

    def guess(self, inputData):

        # The inputData lenght needs to be the same as the first layer lenght (num of neurons)

        if len(inputData) == self.numOfNeurons[0]:
            # Ok
            # First, put the input data on the first neurons layer

            self.neurons[0] = inputData

            # now we have to propagate

            for i in range(self.numOfLayers - 1):
                self.propagate(i)

            # here we only have to return the last layer

            return self.neurons[-1]
        
        else:
            # It's not ok, so we return a false value
            return False

    # now we can train

    def train(self, inputData, expectedGuess):
        
        # The inputData lenght needs to be the same as the first layer lenght (num of neurons)
        # The expectedGuess lenght needs to be the same as the last layer lenght (num of neurons)

        if len(inputData) == self.numOfNeurons[0] and len(expectedGuess) == self.numOfNeurons[-1]:
            # ok
            guess = self.guess(inputData)

            error = []

            for i in range(self.numOfLayers):
                error.append([])
                for j in range(self.numOfNeurons[i]):
                    error[i].append(0)

            # Calculating the error for the last layer

            for i in range(len(guess)):
                error[-1][i] = expectedGuess[i] - guess[i]
            
            # Calculating the error to every layer

            for a in range(self.numOfLayers-2, -1, -1):
                for b in range(self.numOfNeurons[i]):
                    newValue = 0
                    for i in range(self.numOfNeurons[a+1]):
                        denominator = 0
                        for j in range(self.numOfNeurons[a]):
                            denominator += self.weights[a][j][i]
                        newValue += self.weights[a][b][i]*error[a+1][i]/denominator
                    error[a][b] = newValue
            
            # Now we can ajust all the weights

            for i in range(self.numOfLayers-1):
                for j in range(self.numOfNeurons[i]):
                    for k in range(self.numOfNeurons[i+1]):
                        self.weights[i][j][k] += self.learningRate * error[i+1][k] * self.neurons[i][j]

        else:
            # its not ok, lets return a false value
            return False

