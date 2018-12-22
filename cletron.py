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

def sigmoid(value):

    try:

        return 1/(1+math.exp(-value))

    except OverflowError:
        
        return 0
    

def sigmoidDerivate(value):

    return value * (1 - value)

def squareit(value):

    return value**2

def rootit(value):

    if value >= 0:

        return math.sqrt(value)

    else:

        return -math.sqrt(abs(value))

sigmoid = numpy.vectorize(sigmoid)
sigmoidDerivate = numpy.vectorize(sigmoidDerivate)
squareit = numpy.vectorize(squareit)
rootit = numpy.vectorize(rootit)

def randommatrix2d(x, y):

    matrix = []

    for i in range(x):

        matrix.append([])

        for j in range(y):

            matrix[i].append(random.uniform(-1,1))

    return numpy.array(matrix, dtype=numpy.float128)

def randommatrix1d(x):

    matrix = []

    for i in range(x):

        matrix.append(random.uniform(-1,1))

    return numpy.array(matrix, dtype=numpy.float128)

class NeuralNetwork:

    def __init__(self):

        self.learningRate = 0.1
        self.numOfLayers = 3
        self.numOfNeurons = [784, 16, 10]
        self.programName = 'neuralnetwork'

    def generateNeurons(self):

        self.neurons = []
        self.weightedSum = [0]

        for i in range(self.numOfLayers):

            self.neurons.append(numpy.array([0]*self.numOfNeurons[i], dtype=numpy.float128))

            if i > 0:

                self.weightedSum.append(numpy.array([0]*self.numOfNeurons[i], dtype=numpy.float128))

    def generateRandWeights(self):

        self.weights = [0]

        for i in range(1, self.numOfLayers):
            
            matrix = randommatrix2d(self.numOfNeurons[i], self.numOfNeurons[i-1])
            self.weights.append(matrix)

    def storeWeights(self):

        try:

            os.mkdir(self.programName)

        except FileExistsError:

            pass

        for i in range(1, self.numOfLayers):

            numpy.save('{}/weight{}.npy'.format(self.programName, i), self.weights[i])

    def useStoredWeights(self):

        self.weights = [0]

        try:

            for i in range(1, self.numOfLayers):

                weight = numpy.load('{}/weight{}.npy'.format(self.programName, i))
                self.weights.append(weight)

        except FileNotFoundError:
            
            self.generateRandWeights()
            self.storeWeights()

    def storeBias(self):

        try:

            os.mkdir(self.programName)

        except FileExistsError:

            pass

        for i in range(1, self.numOfLayers):

            numpy.save('{}/bias{}.npy'.format(self.programName, i), self.bias[i])

    def useStoredBias(self):

        self.bias = [0]

        try:

            for i in range(1, self.numOfLayers):

                bias = numpy.load('{}/bias{}.npy'.format(self.programName, i))
                self.bias.append(bias)

        except FileNotFoundError:
            
            self.generateRandBias()
            self.storeBias()

    def generateRandBias(self):

        self.bias = [0]
        
        for i in range(1, self.numOfLayers):

            bias = randommatrix1d(self.numOfNeurons[i])
            self.bias.append(bias)

    def propagate(self, layerToPropagate):

        newmatrix = numpy.dot(self.weights[layerToPropagate], self.neurons[layerToPropagate-1])
        newmatrix += self.bias[layerToPropagate]
        self.weightedSum[layerToPropagate] = newmatrix
        newmatrix = sigmoid(newmatrix)
        self.neurons[layerToPropagate] = newmatrix

    def guess(self, inputData):

        if len(inputData) == self.numOfNeurons[0]:

            self.neurons[0] = numpy.array(inputData)

            for i in range(1, self.numOfLayers):

                self.propagate(i)

            return self.neurons[-1]
        
        else:

            return False

    def setTestData(self, inputData, expectedData):

        self.testInputData = inputData
        self.testExpectedData = expectedData

    def setTrainData(self, inputData, expectedData):

        Len = len(inputData)

        if Len <= 60000:
            # Not enough, but you can.
            self.inputGroups = []
            self.inputGroups.append(inputData[:int(Len/2)])
            self.inputGroups.append(inputData[int(Len/2):])
            self.expectedGroups = []
            self.expectedGroups.append(expectedData[:int(Len/2)])
            self.expectedGroups.append(expectedData[int(Len/2):])

        else:
            # We can divide this in groups of 30000, or something like this
            groupsLen = 30000

            while True:
                if Len % groupsLen >= 15000 or Len % groupsLen == 0:
                    # nice!
                    break
                else:
                    groupsLen += 500

            self.inputGroups = []
            self.expectedGroups = []

            for i in range(int(Len / groupsLen)):
                self.inputGroups.append(inputData[groupsLen*i:groupsLen*(i+1)])
                self.expectedGroups.append(expectedData[groupsLen*i:groupsLen*(i+1)])

            self.inputGroups.append(inputData[groupsLen*int(Len / groupsLen):])
            self.expectedGroups.append(expectedData[groupsLen*int(Len / groupsLen):])

    def test(self):

        for i in range(len(self.testInputData)):

            print('Guess:')
            print(self.guess(self.testInputData[i]))
            print('Real answer:')
            print(self.testExpectedData[i])

    def train(self):

        for group in range(len(self.inputGroups)):

            cost = []

            for layer in range(self.numOfLayers):

                cost.append(numpy.array([0]*self.numOfNeurons[layer], dtype=numpy.float64))

            for inputData in range(len(self.inputGroups[group])):

                guess = self.guess(self.inputGroups[group][inputData])
                expectedGuess = numpy.array(self.expectedGroups[group][inputData])

                cost[-1] += squareit(guess - expectedGuess)/len(self.inputGroups[group])

            for L in range(self.numOfLayers-1, 0, -1):

                weightTranspose = self.weights[L].T
                matrix = numpy.dot(weightTranspose, cost[L])
                cost[L-1] = matrix

            for L in range(self.numOfLayers-1, 0, -1):

                weightTranspose = self.weights[L].T
                matrix = numpy.dot(weightTranspose, cost[L])
                cost[L-1] = matrix

                gradient = self.learningRate*numpy.multiply(cost[L], sigmoidDerivate(self.neurons[L]))

                deltaWeight = numpy.dot(gradient, self.neurons[L][numpy.newaxis].T)

                self.weights[L] += deltaWeight

                deltaBias = gradient
                self.bias[L] += deltaBias

        


        


        return guess