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

    def generateNeurons(self): # GERA NEURONIOS ZERADOS

        self.neurons = []
        self.weightedSum = [0]

        for i in range(self.numOfLayers):

            self.neurons.append(numpy.array([0]*self.numOfNeurons[i], dtype=numpy.float128))

            if i > 0:

                self.weightedSum.append(numpy.array([0]*self.numOfNeurons[i], dtype=numpy.float128))

    def generateRandWeights(self): # GERA PESOS ALEATORIOS

        self.weights = [0]

        for i in range(1, self.numOfLayers):
            
            matrix = randommatrix2d(self.numOfNeurons[i], self.numOfNeurons[i-1])
            self.weights.append(matrix)

    def generateRandBias(self): # GERA BIAS ALEATORIOS

        self.bias = [0]
        
        for i in range(1, self.numOfLayers):

            bias = randommatrix1d(self.numOfNeurons[i])
            self.bias.append(bias)

    def storeWeights(self): # GUARDA OS PESOS NUMA PASTA

        try:

            os.mkdir(self.programName)

        except FileExistsError:

            pass

        for i in range(1, self.numOfLayers):

            numpy.save('{}/weight{}.npy'.format(self.programName, i), self.weights[i])

    def storeBias(self): # GUARDA OS BIAS NUMA PASTA

        try:

            os.mkdir(self.programName)

        except FileExistsError:

            pass

        for i in range(1, self.numOfLayers):

            numpy.save('{}/bias{}.npy'.format(self.programName, i), self.bias[i])

    def useStoredWeights(self): # USA OS PESOS GUARDADOS

        self.weights = [0]

        try:

            for i in range(1, self.numOfLayers):

                weight = numpy.load('{}/weight{}.npy'.format(self.programName, i))
                self.weights.append(weight)

        except FileNotFoundError:
            
            self.generateRandWeights()
            self.storeWeights()

    def useStoredBias(self): # USA OS BIAS GUARDADOS

        self.bias = [0]

        try:

            for i in range(1, self.numOfLayers):

                bias = numpy.load('{}/bias{}.npy'.format(self.programName, i))
                self.bias.append(bias)

        except FileNotFoundError:
            
            self.generateRandBias()
            self.storeBias()

    def setTrainData(self, inputData, expectedData): # GUARDA OS DADOS DE TREINAMENTO EM BLOCOS/GRUPOS/SESSOES/EPOCAS

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

    def feedforward(self): # FAZ O FEEDFORWARD

        for layerToPropagate in range(1, self.numOfLayers):

            newmatrix = numpy.dot(self.weights[layerToPropagate], self.neurons[layerToPropagate-1])
            newmatrix += self.bias[layerToPropagate]
            newmatrix = sigmoid(newmatrix)
            self.neurons[layerToPropagate] = newmatrix

    def generateLoss(self): # GERA O LOSS ZERADO

        self.loss = []

        for layer in range(self.numOfLayers):

            self.loss.append(numpy.array([0]*self.numOfNeurons[layer], dtype=numpy.float64))

    def train(self):

        # PASSA POR CADA UM DOS BLOCOS/GRUPOS/SESSOES/EPOCAS
        for group in range(len(self.inputGroups)): 

            # RECRIA O LOSS ZERADO
            self.generateLoss() 

            # PASSA PELOS ELEMENTOS DE CADA BLOCO/GRUPO/SESSAO/EPOCA
            for i in range(len(self.inputGroups[group])):

                # COLOCA O INPUT NA PRIMEIRA CAMADA
                self.neurons[0] = numpy.array(self.inputGroups[group][i])
                # FAZ O FEEDFORWARD
                self.feedforward()
                # RECEBE O RESULTADO DA REDE NEURAL NA ULTIMA CAMADA DELA
                result = self.neurons[-1]
                # GUARDA O RESULTADO ESPERADO EM "expectedResult"
                expectedResult = numpy.array(self.expectedGroups[group][i])

                # AQUI TEMOS UMA PARTE QUE GEROU COMPLICAÇÕES, ENTAO AI VAI O TEXTAO:
                '''
                Como queremos o LOSS MÉDIO DE CADA GRUPO, precisamos somar todos eles, e dividir
                pela quantidade de valores que somamos. Pense assim: Queremos a média de 'a', 'b',
                'c' e 'd'. Podemos somar todos eles e dividir por 4, que seria da forma:

                a + b + c + d
                -------------
                      4
    
                Ou podemos dividir essa conta em 4 frações, ficando da forma:

                 a   +   b   +   c   +   d
                ---     ---     ---     ---
                 4       4       4       4

                 A SEGUNDA FORMA foi a que eu usei nessse caso.

                 Observe que (result - expectedResult)² é o LOSS, mas como vamos somar todos os
                 LOSS desse grupo e dividir pela quantidade de imagens no grupo, temos o
                 len(self.inputGroups[group]) (quantidade de elementos no grupo) dividindo o LOSS.
                '''
                self.loss[-1] += squareit(result - expectedResult)/len(self.inputGroups[group])

                
            # OBSERVE TAMBÉM QUE ATÉ AQUI EU APENAS FIZ O FEEDFORWARD E CALCULEI A MEDIA DO LOSS
                
            # AGORA, COM A MÉDIA DO LOSS, PODEMOS CALCULAR O LOSS PARA OUTRAS CAMADAS,
            # COMO É FEITO AGORA
            # NESSE FOR, O 'L' PASSA POR TODOS OS VALORES DE CAMADA DE TRÁS PRA FRENTE,
            # PRA CALCULAR O LOSS PARA TODAS AS CAMADAS.
            for L in range(self.numOfLayers-1, 0, -1):

                weightTranspose = self.weights[L].T
                self.loss[L-1] = numpy.dot(weightTranspose, self.loss[L])

            # COM O LOSS CALCULADO EM TODAS AS CAMADAS, INICIAMOS O...

            '''
 | |              | |                                          | | (_)            
 | |__   __ _  ___| | ___ __  _ __ ___  _ __   __ _  __ _  __ _| |_ _  ___  _ __  
 | '_ \ / _` |/ __| |/ / '_ \| '__/ _ \| '_ \ / _` |/ _` |/ _` | __| |/ _ \| '_ \ 
 | |_) | (_| | (__|   <| |_) | | | (_) | |_) | (_| | (_| | (_| | |_| | (_) | | | |
 |_.__/ \__,_|\___|_|\_\ .__/|_|  \___/| .__/ \__,_|\__, |\__,_|\__|_|\___/|_| |_|
                       | |             | |           __/ |                        
                       |_|             |_|          |___/     (Backpropagation)

            '''

            # Passamos por todas as camadas de trás pra frente, calculando o gradiente
            for L in range(self.numOfLayers-1, 0, -1):

                # AQUI CALCULAMOS O GRADIENTE
                gradient = self.learningRate*numpy.multiply(self.loss[L], sigmoidDerivate(self.neurons[L]))

                # AQUI MULTIPLICAMOS ELE PELOS NEURONIOS TRANSPOSTOS
                # PARA CALCULAR A VARIAÇÃO DOS PESOS
                deltaWeight = numpy.dot(gradient, self.neurons[L][numpy.newaxis].T)

                # APLICAMOS A VARIAÇÃO DE PESO
                self.weights[L] -= deltaWeight

                # CALCULAMOS A VARIAÇÃO DE BIAS
                deltaBias = gradient
                # APLICAMOS A VARIAÇÃO DE BIAS
                self.bias[L] -= deltaBias

        return result

    def setTestData(self, inputData, expectedData):

        self.testInputData = inputData
        self.testExpectedData = expectedData

    def test(self):

        acertos = 0

        for i in range(len(self.testInputData)):

            print("Network's answer: ")
            # COLOCA O INPUT NA PRIMEIRA CAMADA
            self.neurons[0] = numpy.array(self.testInputData[i])
            # FAZ O FEEDFORWARD
            self.feedforward()
            # RECEBE O RESULTADO DA REDE NEURAL NA ULTIMA CAMADA DELA
            result = self.neurons[-1]
            number = result.tolist().index(max(result))
            print(number)
            print('Real answer:')
            expectednumber = self.testExpectedData[i].index(max(self.testExpectedData[i]))
            print(expectednumber)
            if (number == expectednumber): acertos += 1
            print(number == expectednumber)
            print('-------------------')

        print('Scored {}% of the tests'.format(100*acertos/len(self.testInputData)))