import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1),
                 initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.parameters = [self.initialize_weights(), self.initialize_bias()]

    def initialize_weights(self):
        """
        Initialize weights.
        returns:
            weights: initialized kernel with shape: (kernel_size[0], kernel_size[1], in_channels, out_channels)
        """
        if self.initialize_method == "random":
            return np.random.randn(self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels) * 0.01
        elif self.initialize_method == "xavier":
            xavier_stddev = np.sqrt(2.0 / (self.in_channels + self.out_channels))
            return np.random.randn(self.kernel_size[0], self.kernel_size[1], self.in_channels,
                                   self.out_channels) * xavier_stddev
        elif self.initialize_method == "he":
            he_stddev = np.sqrt(2.0 / self.in_channels)
            return np.random.randn(self.kernel_size[0], self.kernel_size[1], self.in_channels,
                                   self.out_channels) * he_stddev
        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        """
        Initialize bias.
        returns:
            bias: initialized bias with shape: (1, 1, 1, out_channels)
        """
        if self.initialize_method == "random":
            return np.zeros((1, 1, 1, self.out_channels)) * 0.01
        if self.initialize_method == "xavier":
            return np.zeros((1, 1, 1, self.out_channels)) * np.sqrt(1 / self.out_channels)
        if self.initialize_method == "he":
            return np.zeros((1, 1, 1, self.out_channels)) * np.sqrt(2 / self.out_channels)
        else:
            raise ValueError("Invalid initialization method")

    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the convolutional layer.
        args:
            input_shape: shape of the input to the convolutional layer
        returns:
            target_shape: shape of the output of the convolutional layer
        """
        # Assuming 'same' padding and stride of 1
        H = input_shape[0]
        W = input_shape[1]
        num_filters = self.num_filters
        kernel_size = self.kernel_size
        target_shape = (H, W, num_filters)
        return target_shape

    def pad(self, A, padding, pad_value=0):
        """
        Pad the input with zeros.
        args:
            A: input to be padded
            padding: tuple of padding for height and width
            pad_value: value to pad with
        returns:
            A_padded: padded input
        """
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant",
                          constant_values=(pad_value, pad_value))
        return A_padded

    def single_step_convolve(self, a_slic_prev, W, b):
        """
        Convolve a slice of the input with the kernel.
        args:
            a_slic_prev: slice of the input data
            W: kernel
            b: bias
        returns:
            Z: convolved value
        """
        # Element-wise multiplication
        s = np.multiply(a_slic_prev, W)

        # Sum over all elements
        Z = np.sum(s)

        # Add bias as type float using np.float()
        Z = np.float32(Z + b)

        return Z

    def forward(self, A_prev):
        """
        Forward pass for convolutional layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
            returns:
                A: output of the convolutional layer
        """
        # Get the kernel and bias parameters
        W, b = self.get_params()
        # Get the shape of the previous layer activations
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        # Get the shape of the kernel
        (kernel_size_h, kernel_size_w, C_prev, C) = W.shape
        # Get the stride
        stride_h, stride_w = self.stride
        # Get the padding
        padding_h, padding_w = self.padding
        # Calculate the output shape
        H = int((H_prev + 2 * padding_h - kernel_size_h) / stride_h) + 1
        W = int((W_prev + 2 * padding_w - kernel_size_w) / stride_w) + 1
        # Initialize the output activations
        Z = np.zeros((batch_size, H, W, C))
        # Pad the input activations
        A_prev_pad = self.pad(A_prev, self.padding)
        # Perform convolution
        for i in range(batch_size):
            for h in range(H):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h
                for w in range(W):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w
                    for c in range(C):
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        Z[i, h, w, c] = self.single_step_convolve(a_slice_prev, W[..., c], b[..., c])
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for convolutional layer.
        args:
            dZ: gradient of the cost with respect to the output of the convolutional layer
            A_prev: activations from previous layer (or input data)
            A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
        returns:
            dA_prev: gradient of the cost with respect to the input of the convolutional layer
            gradients: list of gradients with respect to the weights and bias
        """
        # Extract parameters
        W, b = self.params
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C_prev, C) = W.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        # Initialize gradients
        dA_prev = np.zeros_like(A_prev)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        # Pad A_prev
        A_prev_pad = self.pad(A_prev, padding_h, padding_w)

        # Pad dA_prev
        dA_prev_pad = self.pad(dA_prev, padding_h, padding_w)

        # Loop over the batch
        for i in range(batch_size):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            # Loop over vertical axis of the output volume
            for h in range(self.output_h):
                # Vertical start and end of the current slice
                h_start = h * stride_h
                h_end = h_start + kernel_size_h

                # Loop over horizontal axis of the output volume
                for w in range(self.output_w):
                    # Horizontal start and end of the current slice
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w

                    # Loop over the channels
                    for c in range(C):
                        # Slice A_prev_pad
                        a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]

                        # Update gradients
                        da_prev_pad[h_start:h_end, w_start:w_end, :] += np.multiply(dZ[i, h, w, c], W[..., c])
                        dW[..., c] += np.multiply(dZ[i, h, w, c], a_slice)
                        db[..., c] += dZ[i, h, w, c]

            # Set the ith example's dA_prev to the unpadded da_prev_pad
            dA_prev[i, :, :, :] = da_prev_pad[padding_h:-padding_h, padding_w:-padding_w, :]

        # Package gradients
        grads = [dW, db]

        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        """
        Update parameters of the convolutional layer.
        args:
            optimizer: optimizer to use for updating parameters
            grads: list of gradients with respect to the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
