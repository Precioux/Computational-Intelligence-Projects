import numpy as np
from NeuralNetworks.code.layers.convolution2d import *

def main():
    # Create a random input image
    A_prev = np.random.randn(2, 3, 3, 1)
    print(f'input: ')
    print(A_prev)

    # Create a Conv2D layer with 1 output channels
    conv_layer = Conv2D(in_channels=3, out_channels=1, name="Conv2DTest", kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1), initialize_method="random")

    # Perform forward pass
    Z = conv_layer.forward(A_prev)
    print(f'Forward:')
    print(Z)

    #Create a fake dZ value
    dZ_fake = np.random.randn(2, 3, 3, 1)

    # Perform backward pass
    dA_prev, grads = conv_layer.backward(dZ_fake, A_prev)

    # Print the derivative of the cost with respect to the activation of the previous layer
    print("dA_prev:")
    print(dA_prev)

    # Print the gradients for the weights and bias
    # print("Gradients:")
    # print(grads)


if __name__ == "__main__":
    main()
