# Cletron

A neural network library made by me, for my own learning, but that can also be used in small projects.

All usage is exemplified in the image recognition program (mnist.go).

## How to use

### Import

First, you need to import the library package. You can do it like this:

```
import "github.com/cl3t0/cletron"
```

### Creation

After that, bring your neural network to life.

```
brain := cletron.NewNeuralNetwork(
  []int{784, 1569, 10}, // Number of neurons per layer
  0.05,                 // Learning rate
)
```

### Configuration

Load the weights and the biases. If you don't have any weights and bias files, don't worry, you can skip this section, because we are going to start the network with random weights and biases

```
brain.UseStoredWeights("./yourFilePath/weightFile.bin")
brain.UseStoredBias("./yourFilePath/biasFile.bin")
```

### Training

Now, start the training!

```
brain.Train(trainImages, trainLabels)
```

### Testing

After that, you can test if your neural network is doing well. :)

```
brain.Test(testImages, testLabels)
```

### Storage

Don't forget to store your data (weights and biases) at the end of the code.

```
brain.StoreWeights("./yourFilePath/weightsFile.bin")
brain.StoreBias("./yourFilePath/biasFile.bin")
```

## Autor

- **Pedro Cleto** - _Library creator_
- **João Retzlaff** - _Collaborator_
