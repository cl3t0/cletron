# neuralnetworkslib
Uma biblioteca de redes neurais feita por mim em Python 3. É bem flexível, mas é iniciante.

O uso dela está todo exemplificado no xor.py.

## Como usar

### Importar

Primeiro é necessário importar a biblioteca. Recomendo importar dessa forma:

```
from nnlib import NeuralNetwork
```

### Criação

Depois crie a sua rede neural:

```
brain = NeuralNetwork()
```

### Configuração

Agora é só configurar a quantidade de camadas e de neurônios em cada camada:

```
brain.numOfLayers = 3
brain.numOfNeurons = [2, 2, 1]
```
E gerar todos os neurônios, pesos e inclinações:

```
brain.generateNeurons()
brain.generateRandWeights()
brain.generateRandBias()
```

### Treinamento

Gere ou crie seus dados de treinamento e treine sua rede:

```
// Vamos treinar 10000 vezes
for i in range(10000):
  brain.train(input, output)
```

### Teste

Depois disso você pode testar se a rede está boa:

```
output = brain.guess(input)
print(output)
```

## Autor

* **Pedro Cleto** - *Library creator*
Vários códigos que podem estar ai, que foram criados usando a minha biblioteca, não são meus.
