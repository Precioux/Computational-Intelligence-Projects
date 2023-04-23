import numpy as np


class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None

    def initialize_weights(self):
        if self.initialize_method == "random":
            # Initialize weights with random values using np.random.randn
            return np.random.randn(self.output_size, self.input_size) * 0.01

        elif self.initialize_method == "xavier":
            # Initialize weights using Xavier initialization
            xavier_stddev = np.sqrt(2 / (self.input_size + self.output_size))
            return np.random.randn(self.output_size, self.input_size) * xavier_stddev

        elif self.initialize_method == "he":
            # Initialize weights using He initialization
            he_stddev = np.sqrt(2 / self.input_size)
            return np.random.randn(self.output_size, self.input_size) * he_stddev

        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        # TODO: Initialize bias with zeros
        return np.zeros((None, 1))

    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size)
            returns:
                Z: output of the fully connected layer
        """
        # NOTICE: BATCH_SIZE is the first dimension of A_prev
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        # TODO: Implement forward pass for fully connected layer
        if None:  # check if A_prev is output of convolutional layer
            batch_size = None
            A_prev_tmp = A_prev_tmp.reshape(None, -1).T
        self.reshaped_shape = A_prev_tmp.shape

        # TODO: Forward part
        W, b = None
        Z = None @ None + None
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        if None:  # check if A_prev is output of convolutional layer
            batch_size = None
            A_prev_tmp = A_prev_tmp.reshape(None, -1).T

        # TODO: backward part
        W, b = None
        dW = None @ None.T / None
        db = np.sum(None, axis=1, keepdims=True) / None
        dA_prev = None.T @ None
        grads = [dW, db]
        # reshape dA_prev to the shape of A_prev
        if None:    # check if A_prev is output of convolutional layer
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
