from nnlib import NeuralNetwork
import random

brain = NeuralNetwork()

brain.numOfLayers = 3
brain.numOfNeurons = [2, 2, 1]

brain.generateNeurons()
brain.generateRandWeights()
brain.generateRandBias()

trainingData = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# Training...
for i in range(100000):
    q = random.randint(0, 3)
    brain.train([trainingData[q][0], trainingData[q][1]], [trainingData[q][2]])

for i in trainingData:
    guess = brain.guess([i[0], i[1]])
    answer = i[2]
    print("Guess: " + str(guess[0]))
    print("Real Answer: " + str(answer))
