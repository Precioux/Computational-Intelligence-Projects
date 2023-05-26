import numpy as np
from NeuralNetworks.code.layers import maxpooling2d


def test_maxpool():
    # Create an instance of MaxPool2D layer
    maxpool_layer = maxpooling2d.MaxPool2D(kernel_size=2, stride=1, mode="max")

    # Generate random input data
    A_prev = np.random.randn(4, 6, 6, 3)

    # Perform forward pass
    Z = maxpool_layer.forward(A_prev)

    # Print the input and output shapes
    print("Input shape:", A_prev.shape)
    print("Output shape:", Z.shape)

    # Perform backward pass with dummy gradients
    dA = np.ones_like(Z)
    dA_prev, _ = maxpool_layer.backward(dA, A_prev)

    # Print the gradients with respect to input
    print("Gradients with respect to input shape:", dA_prev.shape)


if __name__ == "__main__":
    test_maxpool()
