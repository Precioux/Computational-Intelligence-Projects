import numpy as np


class MaxPool2D:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        """
        Max pooling layer.
            args:
                kernel_size: size of the kernel
                stride: stride of the kernel
                mode: max or average
            Question:Why we don't need to set name for the layer?
            Ans:
            The name parameter is not included in the constructor because it is
            not a necessary property for the functioning of the max pooling layer.
            The name parameter is commonly used when building complex neural network
            architectures or when you need to uniquely identify a specific layer
            within a network. However, for a basic implementation of a max pooling layer,
            it is not essential to assign a name to it.
        """
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode

    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the layer.
            args:
                input_shape: shape of the input
            returns:
                output_shape: shape of the output
        """
        H = (input_shape[0] - self.kernel_size[0]) // self.stride[0] + 1
        W = (input_shape[1] - self.kernel_size[1]) // self.stride[1] + 1
        return H, W

    def forward(self, A_prev):
        """
        Forward pass for max pooling layer.
            args:
                A_prev: activations from previous layer (or input data)
            returns:
                A: output of the max pooling layer
        """
        # Get dimensions of input
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        # Get dimensions of filter
        (f_h, f_w) = self.kernel_size
        # Get stride values
        strideh, stridew = self.stride
        # Compute output dimensions
        H = int((H_prev - f_h) / strideh) + 1
        W = int((W_prev - f_w) / stridew) + 1
        # Initialize output with zeros
        A = np.zeros((batch_size, H, W, C_prev))
        # Loop over each training example
        for i in range(batch_size):
            # Loop over vertical axis of output volume
            for h in range(H):
                h_start = h * strideh
                h_end = h_start + f_h
                # Loop over horizontal axis of output volume
                for w in range(W):
                    w_start = w * stridew
                    w_end = w_start + f_w
                    # Loop over channels of output volume
                    for c in range(C_prev):
                        # Slice input for current filter
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            # Compute max value for filter
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            # Compute average value for filter
                            A[i, h, w, c] = np.mean(a_prev_slice)
                        else:
                            raise ValueError("Invalid mode")

        return A

    def create_mask_from_window(self, x):
        """
        Create a mask from an input matrix x, to identify the max entry of x.
            args:
                x: numpy array
            returns:
                mask: numpy array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        mask = x == np.max(x)
        return mask

    def distribute_value(self, dz, shape):
        """
        Distribute the input value in the matrix of dimension shape.
            args:
                dz: input scalar
                shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
            returns:
                a: distributed value
        """
        # Implement distribute_value
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        return a

    def backward(self, dZ, A_prev):
        """
        Backward pass for max pooling layer.
            args:
                dA: gradient of cost with respect to the output of the max pooling layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: gradient of cost with respect to the input of the max pooling layer
        """
        # Implement backward pass for max pooling layer
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        batch_size, H_prev, W_prev, C_prev = A_prev.shape
        batch_size, H, W, C = dZ.shape
        dA_prev = np.zeros((batch_size, H_prev, W_prev, C_prev))
        for i in range(batch_size):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        h_start = h * strideh
                        h_end = h_start + f_h
                        w_start = w * stridew
                        w_end = w_start + f_w
                        if self.mode == "max":
                            a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, :]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, h_start:h_end, w_start:w_end, :] += np.multiply(mask, dZ[i, h, w, c])
                        elif self.mode == "average":
                            dz = dZ[i, h, w, c]
                            dA_prev[i, h_start:h_end, w_start:w_end, :] += self.distribute_value(dz, (f_h, f_w))
                        else:
                            raise ValueError("Invalid mode")
        # Don't change the return
        return dA_prev, None
