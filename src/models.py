import numpy as np


class NeuralNetwork():

    def __init__(self, n_input, n_output, hidden_sizes):
        sizes = [n_input] + hidden_sizes + [n_output]
        self.weight_mean = 0
        self.std = 2
        self.weights = []
        self.bias = []
        self.activations = []
        for i in range(1, len(sizes)):
            self.weights.append(np.random.normal(
                loc=self.weight_mean, scale=self.std, size=(sizes[i], sizes[i-1])))
            self.bias.append(np.random.normal(
                loc=self.weight_mean, scale=self.std, size=sizes[i]))
            self.activations.append(NeuralNetwork.sigmoid)
        self.activations[-1] = NeuralNetwork.sigmoid

    def create(weights, bias, activations):
        a = NeuralNetwork(0, 0, [1])
        a.weights = [np.copy(x) for x in weights]
        a.bias = [np.copy(x) for x in bias]
        a.activations = [x for x in activations]
        return a

    def copy(self):
        return NeuralNetwork.create(self.weights, self.bias, self.activations)

    def default():
        return NeuralNetwork(8, 3, [8, 6, 6])

    def relu(x):
        return x * (x > 0)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def linear(x):
        return x

    def forward(self, x: np.array):
        out = np.copy(x)
        for i, weight in enumerate(self.weights):
            out = weight @ out
            out = out + self.bias[i]
            out = self.activations[i](out)
        return out

    def mutateLayer(self, layer_i, n):
        layer_shape = self.weights[layer_i].shape
        n_weights = self.weights[layer_i].size
        mutated = []
        i = 0
        while i < n:
            weight_i = np.random.randint(0, n_weights)
            if weight_i not in mutated:
                mutated.append(weight_i)
                new_weight = np.random.normal(
                    loc=self.weight_mean, scale=self.std, size=1)[0]
                new_bias = np.random.normal(
                    loc=self.weight_mean, scale=self.std, size=1)[0]

                self.weights[layer_i][weight_i//layer_shape[1],
                                      weight_i % layer_shape[1]] = new_weight
                self.bias[layer_i][weight_i//layer_shape[1]] = new_bias
            i += 1

    def mutate(self, layer_p: float, n_mutated):
        mutate_l_i = np.random.randint(0, len(self.weights))
        self.mutateLayer(mutate_l_i, n_mutated)
        for layer_i in range(len(self.weights)):
            if layer_i != mutate_l_i and np.random.random() < layer_p:
                self.mutateLayer(layer_i, n_mutated)

    def cross(self, b):
        new = self.copy()
        layer_i = np.random.randint(0, len(new.weights))
        new.weights[layer_i] = np.copy(b.weights[layer_i])
        new.bias[layer_i] = np.copy(b.bias[layer_i])
        new.mutate(0.3, 2)
        return new

    def load(path, n_layers=3):
        container = np.load(path)
        data = [container[x] for x in container]
        bias = data[n_layers+1:]
        weights = data[:n_layers+1]
        activations = [NeuralNetwork.sigmoid for x in range(len(weights))]
        activations[-1] = NeuralNetwork.sigmoid
        return NeuralNetwork.create(weights, bias, activations)

    def save(self, path):
        data = self.weights + [x for x in self.bias]
        np.savez(path, *data)