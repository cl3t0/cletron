from mnist import MNIST
from cletron import NeuralNetwork

brain = NeuralNetwork()

brain.numOfLayers = 3
brain.numOfNeurons = [784, 1569, 10]
brain.learningRate = 0.05
brain.programName = 'image-recognition'

brain.generateNeurons()
brain.useStoredWeights()
brain.useStoredBias()


mndata = MNIST('image-recognition_trainingdata')

print('Loading training data...')

train_images, train_labels_orig = mndata.load_training()

print('Loaded!!')

quantidade = len(train_images)

print('--- applying the image process ---')
for i in range(quantidade):
	for j in range(len(train_images[i])):
		train_images[i][j] = train_images[i][j]/255
print('--- process ended ---')

print('--- applying the label process --')
train_labels = []
for i in range(quantidade):
	result = [0]*10
	result[train_labels_orig[i]] = 1
	train_labels.append(result)
print('--- process ended ---')

brain.setTrainData(train_images, train_labels)

print('Starting to train...')

brain.train()

print('Training ended!!')

print('Loading testing data...')

test_images, test_labels_orig = mndata.load_testing()

print('Loaded!!')

quantidade = len(test_images)

print('--- applying the image process ---')
for i in range(quantidade):
	for j in range(len(test_images[i])):
		test_images[i][j] = test_images[i][j]/255
print('--- process ended ---')

print('--- applying the label process --')
test_labels = []
for i in range(quantidade):
	result = [0]*10
	result[test_labels_orig[i]] = 1
	test_labels.append(result)
print('--- process ended ---')

brain.setTestData(test_images, test_labels)

print('Starting to test...')

brain.test()

print('Test ended!!')

brain.storeWeights()
brain.storeBias()
