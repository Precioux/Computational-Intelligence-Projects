import numpy as np


class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "he"):
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
            # Initialize weights using he initialization
            he_stddev = np.sqrt(2 / self.input_size)
            return np.random.randn(self.output_size, self.input_size) * he_stddev

        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        # Initialize bias with zeros
        return np.zeros((self.output_size, 1))

    def forward(self, A_prev):
        print('FC Forward Step:')
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)
        print(f'input shape: {self.input_shape}')

        if len(A_prev_tmp.shape) > 2:
            print('yes to conv')
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1)

        else:
            batch_size = A_prev_tmp.shape[0]

        self.reshaped_shape = A_prev_tmp.shape

        # Forward part
        W, b = self.parameters

        # Add each item of the dot product to the bias
        bias_expanded = np.tile(b.T, (batch_size, 1))
        Z = np.dot(W, A_prev_tmp.T).T
        Z_with_bias = Z + bias_expanded
        print('FC forward done!')

        return Z_with_bias

    def backward(self, dZ, A_prev):
        print('FC Backward Step:')
        A_prev_tmp = np.copy(A_prev)

        if len(A_prev_tmp.shape) > 2:
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1)
        else:
            batch_size = A_prev_tmp.shape[0]

        # Backward part
        W, b = self.parameters
        dW = np.dot(dZ.T, A_prev_tmp) / batch_size
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dA_prev = np.dot(dZ, W)
        grads = [dW, db]
        if self.input_shape is not None:
            print('YES TO CONV')
            dA_prev = dA_prev.reshape(self.input_shape)
        print('FC backward done!')
        return dA_prev, grads

    def update(self, optimizer, grads, epoch):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
                epoch : current epoch
        """
        g = {}
        g[0] = grads[0]
        g[1] = grads[1].T
        self.parameters = optimizer.update(g, self.name, epoch)
