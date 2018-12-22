# Cletron
A neural network library made by me, for my own learning, but that can be used in small projects.

All usage is exemplified in the image recognition program (image-recognition.py).

## how to use

### Import

First, you need to import the library. To do this, you can do as follow.

```
from cletron import NeuralNetwork
```

### Criation

After that, bring your neural network to life.

```
brain = NeuralNetwork()
```

### Configuration

Now you have to write some lines to configure the number of layers, the number of neurons in each layer and the learning rate.

```
brain.numOfLayers = 3
brain.numOfNeurons = [784, 1569, 10]
brain.learningRate = 0.05
```

Name your neural network:

```
brain.programName = 'image-recognition'
```

Generate the neurons:

```
brain.generateNeurons()
```

Now, load the weights and the biases. (Relax, if these files do not exist, we will create then for you. :D)

```
brain.useStoredWeights()
brain.useStoredBias()
```

### Training

Set your training data.
```
brain.setTrainData(train_images, train_labels)
```

Now, start the training!
```
brain.train()
```
### Test

After that, you can test if your neural network are doing well. :)
Set your testing data.
```
brain.setTestData(test_images, test_labels)
```
Now, start the testing!
```
brain.test()
```
### Storage

Don't forget to store your data (weights and biases) at the end of the code.

```
brain.storeWeights()
brain.storeBias()
```

## Autor

* **Pedro Cleto** - *Library creator*