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

    def showArq(self):
        print('--------------------------------------------------')
        print('|  Layer 0  |       {}       |       {}       |'.format(round(self.neurons[0].item(0), 1), round(self.neurons[0].item(1), 1)))
        print('| Weights 0 |0-{}-0|0-{}-1|0-{}-2|1-{}-0|1-{}-2|2-{}-2|'.format(round(self.weights[1].item((0, 0)), 1), round(self.weights[1].item((1, 0)), 1), round(self.weights[1].item((2, 0)), 1), round(self.weights[1].item((0, 1)), 1), round(self.weights[1].item((1, 1)), 1), round(self.weights[1].item((2, 1)), 1)))
        print('|  Layer 1  |     {}     |     {}     |     {}     |'.format(round(self.neurons[1].item(0), 1), round(self.neurons[1].item(1), 1), round(self.neurons[1].item(2), 1)))
        print('| Weights 0 | 0-{}-0 | 1-{}-0 | 2-{}-0 |'.format(round(self.weights[2].item((0, 0)), 1), round(self.weights[2].item((0, 1)), 1), round(self.weights[2].item((0, 2)), 1)))
        print('|  Layer 2  |     {}     |'.format(round(self.neurons[2].item(0), 1)))
        print('--------------------------------------------------')

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

        print('PROPAGANDO: {}->{}'.format(layerToPropagate-1, layerToPropagate))

        newmatrix = numpy.dot(self.weights[layerToPropagate], self.neurons[layerToPropagate-1])
        newmatrix += self.bias[layerToPropagate]
        self.weightedSum[layerToPropagate] = newmatrix
        newmatrix = sigmoid(newmatrix)
        self.neurons[layerToPropagate] = newmatrix

        self.showArq()

    def guess(self, inputData):

        if len(inputData) == self.numOfNeurons[0]:

            self.showArq()

            print('INSERINDO O INPUT...')

            self.neurons[0] = numpy.array(inputData)

            self.showArq()

            print('INICIANDO PROPAGAÇÃO...')

            for i in range(1, self.numOfLayers):

                self.propagate(i)

            print('ENCERRANDO PROPAGAÇÃO...')

            print('OUTPUT RETORNADO!')

            return self.neurons[-1]
        
        else:

            return False

    def train(self, inputData, expectedGuess):
        
        if len(inputData) == self.numOfNeurons[0] and len(expectedGuess) == self.numOfNeurons[-1]:

            guess = self.guess(inputData)

            expectedGuess = numpy.array(expectedGuess)

            cost = []

            for i in range(self.numOfLayers):

                cost.append(numpy.array([0]*self.numOfNeurons[i], dtype=numpy.float64))

            cost[-1] = squareit(guess - expectedGuess)

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

        else:
            return False