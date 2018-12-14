# Neural Networks Library (nnlib)
Uma biblioteca de redes neurais feita por mim em Python 3. É iniciante, mas é bem flexível.

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

Dar nome ao seu programa:

```
brain.programName = 'nomedomeuprograma'
```

Gerar todos os neurônios:

```
brain.generateNeurons()
```

E carregar todos os *weights* e os *bias* (relaxe, se não existir um arquivo para guardar os *weights* e os *bias*, ele irá criar para você :D):

```
brain.useStoredWeights()
brain.useStoredBias()
```

### Treinamento

Gere ou crie seus dados de treinamento e treine sua rede (Não se esqueça de treinar diversas vezes, para que o resultado se torne mais acurado):

```
brain.train(input, output)
```

### Teste

Depois disso você pode testar se a rede está com bons resultados:

```
output = brain.guess(input)
print(output)
```

### Armazenamento

Não se esqueça de armazenar os *weights* e os *bias* ao final do código.

```
brain.storeWeights()
brain.storeBias()
```

## Autor

* **Pedro Cleto** - *Library creator*

Vários códigos que podem estar ai, que foram criados usando a minha biblioteca, não necessariamente são meus.
