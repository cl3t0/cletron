from mnist import MNIST
from nnlib import NeuralNetwork

brain = NeuralNetwork()

brain.numOfLayers = 4
brain.numOfNeurons = [784, 10, 10, 10]
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
	if i < 10:
		print(result)
	print('Treinando com a imagem {}/{}'.format(i, quantidade))
	brain.train(train_images[i], result)

test_images, test_labels = mndata.load_testing()

for i in range(len(test_labels)):
	brain.guess(test_images[i])
	print()

brain.storeWeights()
brain.storeBias()