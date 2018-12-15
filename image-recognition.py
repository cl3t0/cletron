from mnist import MNIST
from nnlib import NeuralNetwork

brain = NeuralNetwork()

brain.numOfLayers = 3
brain.numOfNeurons = [784, 16, 10]
brain.programName = 'image-recognition'

brain.generateNeurons()
brain.useStoredWeights()
brain.useStoredBias()


mndata = MNIST('image-recognition_trainingdata')

print('Carregando os dados para o treinamento (vai demorar bastante)')

train_images, train_labels = mndata.load_training()

print('Carregado!!')
print('O treinamento vai iniciar!')

quantidade = len(train_labels)

for i in range(quantidade):
	result = [0]*10
	result[train_labels[i]] = 1
	brain.train(train_images[i], result)
	if i % 1000 == 0: print(round(100*i/quantidade, 1))


test_images, test_labels = mndata.load_testing()

for i in range(1000):
	guess = brain.guess(test_images[i])
	print('------------')
	print(guess)
	print('guess: ' + str(guess.tolist().index(max(guess))))
	print('real answer: ' + str(test_labels[i]))

brain.storeWeights()
brain.storeBias()