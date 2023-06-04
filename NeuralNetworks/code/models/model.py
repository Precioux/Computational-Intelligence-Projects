from NeuralNetworks.code.layers.convolution2d import *
from NeuralNetworks.code.layers.maxpooling2d import MaxPool2D
from NeuralNetworks.code.layers.fullyconnected import FC
from NeuralNetworks.code.activations.activations import Activation, get_activation
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)

    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        return isinstance(layer, (Conv2D, MaxPool2D, FC))

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        return isinstance(layer, Activation)

    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x  # input
        for l in range(len(self.layers_names)):
            if self.is_layer(self.layers_names[l]):
                A = self.model[self.layers_names[l]].activation.forward(A)
            if self.is_activation(self.layers_names[l]):
                Z = self.model[self.layers_names[l]].forward(A)

        return tmp

    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        for l in reversed(range(len(self.layers_names))):
            if l > 1:
                Z, A = tmp[l * 2 - 1], tmp[l * 2 - 2]
            else:
                Z, A = tmp[l * 2 - 1], x
            dZ = dA * self.model[self.layers_names[l]].activation.backward(Z)
            dA, grad = self.model[self.layers_names[l]].backward(dZ, A)
            grads[self.layers_names[l]] = grad
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for layer_name in self.layers_names:
            if self.is_layer(self.model[layer_name]) and not isinstance(self.model[layer_name], MaxPool2D):
                self.model[layer_name].update(grads[layer_name])

    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.compute(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss

    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)

    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)

    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        last_index = min(index + batch_size, len(order))
        batch = order[index:last_index]
        print(f'Batch is {batch}')
        if X.ndim == 4:
            bx = X[batch]
            by = []
            for e in batch:
                counter = 0
                for a in y:
                    if counter == e:
                        by.append(a)
                    counter = counter + 1
            print('fine1')
            return bx, by
        else:
            bx = X[batch, :]
            by = []
            for e in batch:
                counter = 0
                for a in y:
                    if counter == e:
                        by.append(a)
                    counter = counter + 1
            print('fine2')
            return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        m = X.shape[0] if X.ndim == 4 else X.shape[1]
        order = self.shuffle(m, False)
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b * batch_size, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion.compute(AL, by)
        return cost

    def train(self, X, y, epochs, val=None, batch_size=3, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        print("#################################################3")
        train_cost = []
        val_cost = []
        m = 0  # number of samples
        if X.ndim == 4:  # image [batch_size, height, width, channels]
            m = X.shape[0]
        else:  # data [samples, features]
            m = X.shape[1]

        for e in tqdm(range(1, epochs + 1)):
            # generate a random order of indices for shuffling the training data. to introduce randomness and prevent
            # any potential bias that may arise due to the order of samples in the dataset
            order = self.shuffle(m, shuffling)
            print(f'order : {order}')
            cost = 0
            print(f'times : {m // batch_size}')
            for b in range(m // batch_size):
                print(f'b is {b}')
                bx, by = self.batch(X, y, batch_size, b * batch_size, order)
                print(f'bx = {bx}')
                print(f'by = {by}')
                cost += self.one_epoch(bx, by)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.compute_loss(val[0], val[1], batch_size))
            if verbose != 0:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost

    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        return self.forward(X)[-1]
