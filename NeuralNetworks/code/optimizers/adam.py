import numpy as np

class Adam:
    def __init__(self, layers_dict, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = {name: layer for name, layer in layers_dict.items() if hasattr(layer, 'parameters')}
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for name, layer in self.layers.items():
            v = [np.zeros_like(p) for p in layer.parameters]
            s = [np.zeros_like(p) for p in layer.parameters]
            self.V[name] = v
            self.S[name] = s

    def update(self, grads, name, epoch):
        print('IN ADAM')
        layer = self.layers[name]
        params = []
        for i in range(len(grads)):
            self.V[name][i] = self.beta1 * self.V[name][i] + (1 - self.beta1) * grads[i]
            self.S[name][i] = self.beta2 * self.S[name][i] + (1 - self.beta2) * np.square(grads[i])
            V_corrected = self.V[name][i] / (1 - np.power(self.beta1, epoch))
            S_corrected = self.S[name][i] / (1 - np.power(self.beta2, epoch))
            params.append(
                layer.parameters[i] - self.learning_rate * V_corrected / (np.sqrt(S_corrected) + self.epsilon))
        return params
