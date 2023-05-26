import numpy as np
from NeuralNetworks.code.activations.activations import *


def test_activations():
    # Create instances of the activation functions
    sigmoid_activation = Sigmoid()
    relu_activation = ReLU()
    linear_activation = LinearActivation()
    tanh_activation = Tanh()

    # Generate random input
    Z = np.random.randn(4, 6)

    # Perform forward pass for each activation function
    A_sigmoid = sigmoid_activation.forward(Z)
    A_relu = relu_activation.forward(Z)
    A_linear = linear_activation.forward(Z)
    A_tanh = tanh_activation.forward(Z)

    # Generate random dA values for backward pass
    dA = np.random.randn(*A_sigmoid.shape)

    # Perform backward pass for each activation function
    B_sigmoid = sigmoid_activation.backward(dA, Z)
    B_relu = relu_activation.backward(dA, Z)
    B_linear = linear_activation.backward(dA,Z)
    B_tanh = tanh_activation.backward(dA, Z)

    # Print the input and output shapes for each activation function
    print('input:')
    print(Z)
    print("Input shape:", Z.shape)
    print("Sigmoid output shape:", A_sigmoid.shape)
    print("ReLU output shape:", A_relu.shape)
    print("Linear output shape:", A_linear.shape)
    print("Tanh output shape:", A_tanh.shape)
    print('forward result:')
    print("Sigmoid:", A_sigmoid)
    print("ReLU:", A_relu)
    print("Linear:", A_linear)
    print("Tanh:", A_tanh)
    print('backward result:')
    print("Sigmoid:", B_sigmoid)
    print("ReLU:", B_relu)
    print("Linear:", B_linear)
    print("Tanh:", B_tanh)


if __name__ == "__main__":
    test_activations()
