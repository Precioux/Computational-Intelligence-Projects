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
        # print('Weights:')
        # print(self.parameters[0].shape)
        # print(self.parameters[0])
        # print('Bias:')
        # print(self.parameters[1].shape)
        # print(self.parameters[1])

    def initialize_weights(self):
        # Initialize weights
        kernel_shape = (self.kernel_size[0] // self.stride[0], self.kernel_size[1] // self.stride[1], self.in_channels,
                        self.out_channels)
        if self.initialize_method == "random":
            return np.random.randn(*kernel_shape) * 0.01
        elif self.initialize_method == "xavier":
            xavier_stddev = np.sqrt(2.0 / (self.in_channels + self.out_channels))
            return np.random.randn(*kernel_shape) * xavier_stddev
        elif self.initialize_method == "he":
            he_stddev = np.sqrt(2.0 / self.in_channels)
            return np.random.randn(*kernel_shape) * he_stddev
        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        # Initialize bias
        bias_shape = (1, 1, 1, self.out_channels)
        if self.initialize_method == "random":
            return np.zeros(bias_shape)
        elif self.initialize_method == "xavier":
            return np.zeros(bias_shape) * np.sqrt(1 / self.out_channels)
        elif self.initialize_method == "he":
            return np.zeros(bias_shape) * np.sqrt(2 / self.out_channels)
        else:
            raise ValueError("Invalid initialization method")

    def target_shape(self, input_shape):
        H = input_shape[0]
        W = input_shape[1]
        num_filters = self.out_channels
        target_shape = (H, W, num_filters)
        return target_shape

    def pad(self, A, padding, pad_value=0):
        return np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
                      mode="constant", constant_values=(pad_value, pad_value))

    def single_step_convolve(self, a_slice_prev, W, b):
        s = np.multiply(a_slice_prev, W)
        Z = np.sum(s)
        Z = np.float32(Z + b)
        return Z

    def forward(self, A_prev):
        # Get the parameters of the CNN.
        W, b = self.parameters
        print('IN CNN FORWARD')
        # print(f'W : {W.shape}')
        # print(f'b : {b.shape}')

        # Get the shape of the input.
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        # print(f'input : {A_prev.shape}')
        # print(f'batch_size :{batch_size}')
        # print(f'H_prev : {H_prev}')
        # print(f'W_prev : {W_prev}')
        # print(f'C_prev : {C_prev}')

        # Get the shape of the kernel.
        (kernel_size_h, kernel_size_w, _, C) = W.shape

        # Get the stride and padding.
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        # Calculate the output shape.
        Height = int((H_prev + 2 * padding_h - kernel_size_h) / stride_h) + 1
        Width = int((W_prev + 2 * padding_w - kernel_size_w) / stride_w) + 1

        # Initialize the output matrix.
        Z = np.zeros((batch_size, Height, Width, C))

        # Pad the input.
        A_prev_pad = self.pad(A_prev, self.padding)

        # Iterate over the batch size, the height, the width, and the number of channels.
        for i in range(batch_size):
            for h in range(Height):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h
                for w in range(Width):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w
                    for c in range(C):
                        # Get the current patch of the input.
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]

                        # Get the weight matrix for the current channel.
                        W_current_channel = W[:, :, :, c]

                        # Get the bias for the current channel.
                        b_current_channel = b[0, 0, 0, c]

                        # Calculate the convolution of the current patch of the input with the current kernel.
                        Z[i, h, w, c] = self.single_step_convolve(a_slice_prev, W_current_channel, b_current_channel)

        # Return the output matrix.
        print(f'Z : {Z.shape}')
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
        print('IN CNN BACKWARD')
        print(f'dZ : {dZ.shape}')
        print(f'A_prev : {A_prev.shape}')
        W, b = self.parameters
        print(f'b : {b.shape}')
        print(f'W : {W.shape}')
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, _, C) = W.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        Height, Width = int((H_prev + 2 * padding_h - kernel_size_h) / stride_h) + 1, int(
            (W_prev + 2 * padding_w - kernel_size_w) / stride_w) + 1
        dA_prev = np.zeros_like(A_prev,dtype='float64')
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        print(f'Height :{Height}')
        print(f'Width : {Width}')
        print(f'batch_size : {batch_size}')

        # Pad the input tensor with zeros.
        A_prev_pad = np.pad(A_prev, ((0, 0), (padding_h, padding_h), (padding_w, padding_w), (0, 0)), mode="constant",
                            constant_values=(0, 0))
        # Calculate the gradient of the cost with respect to the input tensor.
        for i in range(batch_size):
            for h in range(dZ.shape[2]):
                for w in range(dZ.shape[1]):
                    for c in range(C):
                        # Calculate the index of the current output pixel in the original input tensor.
                        h_start = h * stride_h
                        h_end = h_start + kernel_size_h
                        w_start = w * stride_w
                        w_end = w_start + kernel_size_w
                        a_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]

                        # Calculate the gradient of the cost with respect to the current input pixel.
                        dA_prev[i, h_start:h_end, w_start:w_end, :] += dZ[i, h, w, c] * W[:, :, :, c]

                        # Update the gradient of the weights.
                        dW[:, :, :, c] += dZ[i, h, w, c] * a_slice

                        # Update the gradient of the bias.
                        db[:, :, :, c] += dZ[i, h, w, c]
        grads = [dW, db]
        print('CNN BACKWARD DONE')
        return dA_prev, grads

    def update(self, optimizer, grads,epoch):
        g ={}
        g[0] = grads[0]
        g[1] = grads[1].T
        self.parameters = optimizer.update(g, self.name,epoch)
