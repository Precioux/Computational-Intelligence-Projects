import numpy as np
from ..layers.fullyconnected import FC

def main():
    # Create an instance of the FC class with ReLU activation
    fc_layer = FC(input_size=3, output_size=2, name="FCtest", initialize_method="random")

    # Generate random input data
    A_prev = np.random.randn(4, 3)

    # Print the input
    print("Input:")
    print(A_prev)

    # Perform forward pass
    Z = fc_layer.forward(A_prev)

    # Perform backward pass with a fake dZ value
    dZ_fake = np.random.randn(4, 2)
    dA_prev, grads = fc_layer.backward(dZ_fake, A_prev)

    # Print the derivative of the cost with respect to the activation of the previous layer
    print("dA_prev:")
    print(dA_prev)

    # Print the gradients for the weights and bias
    print("Gradients:")
    print(grads)


if __name__ == "__main__":
    main()
