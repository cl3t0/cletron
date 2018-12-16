#!/usr/bin/env python3

from mnist import MNIST
from nnlib import NeuralNetwork

brain = NeuralNetwork()

brain.numOfLayers = 4
brain.numOfNeurons = [784, 32, 16, 10]
brain.programName = 'image-recognition'

brain.generateNeurons()
brain.useStoredWeights()
brain.useStoredBias()


mndata = MNIST('image-recognition_trainingdata')

print('Carregando os dados para o treinamento...')

train_images, train_labels = mndata.load_training()

print('Carregado!!')

print('--- tratando as imagens ---')
for i in range(len(train_images)):
	for j in range(len(train_images[i])):
		train_images[i][j] = train_images[i][j]/255
print('--- imagens tratadas ---')

print('O treinamento iniciado!')

quantidade = len(train_labels)

for i in range(quantidade):
	result = [0]*10
	result[train_labels[i]] = 1
	brain.train(train_images[i], result)
	if i % 1000 == 0: print(round(100*i/quantidade, 1))

print('Treinamento finalizado!!')

print('Carregando os dados para o teste...')

test_images, test_labels = mndata.load_testing()

print('Dados carregados!')

print('--- tratando as imagens ---')
for i in range(len(test_images)):
	for j in range(len(test_images[i])):
		test_images[i][j] = test_images[i][j]/255
print('--- imagens tratadas ---')

print('Teste iniciado!')

for i in range(1000):
	guess = brain.guess(test_images[i])
	print('------------')
	print(guess)
	print('guess: ' + str(guess.tolist().index(max(guess))))
	print('real answer: ' + str(test_labels[i]))

print('Teste finalizado!')

brain.storeWeights()
brain.storeBias()
