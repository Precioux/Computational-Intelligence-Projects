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
        # Initialize weights
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels)
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
        W, b = self.parameters
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C_prev, C) = W.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        H = int((H_prev + 2 * padding_h - kernel_size_h) / stride_h) + 1
        W = int((W_prev + 2 * padding_w - kernel_size_w) / stride_w) + 1
        Z = np.zeros((batch_size, H, W, C))
        A_prev_pad = self.pad(A_prev, self.padding)
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
        W, b = self.parameters
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C_prev, C) = W.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        dA_prev = np.zeros_like(A_prev)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        A_prev_pad = self.pad(A_prev, (padding_h, padding_w))

        for i in range(batch_size):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev[i]

            for h in range(self.output_h):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h

                for w in range(self.output_w):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w

                    for c in range(C):
                        a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]
                        da_prev_pad[h_start:h_end, w_start:w_end, :] += np.multiply(dZ[i, h, w, c], W[..., c])
                        dW[..., c] += np.multiply(dZ[i, h, w, c], a_slice)
                        db[..., c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[padding_h:-padding_h, padding_w:-padding_w, :]

        grads = [dW, db]
        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)
