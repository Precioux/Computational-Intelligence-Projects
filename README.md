# Computational Intelligence Course Projects

This repository contains three projects developed as part of the Computational Intelligence course. Each project focuses on different aspects of computational intelligence techniques. Below is a brief overview of each project:
- Amirkabir University of Technology
- Spring 2023
  
## Project 1: Neural Network Implementation

In this project, a neural network is implemented from scratch, incorporating various layers such as Fully Connected (FC), Conv2D, and Maxpooling. The implementation includes popular optimizers like Adam and Gradient Descent, as well as loss functions including Binary Cross Entropy and Mean Squared Error. Different activation functions such as Sigmoid, ReLU, and Linear are also implemented.

The neural network model is utilized to accomplish two tasks:

1. **California Housing Dataset**: The model is trained to predict housing prices in California based on various features.
2. **MNIST Dataset**: The model is trained to classify and label the digits 2 and 5 in the MNIST dataset.

## Project 2: Fuzzy Car Controller

This project focuses on implementing a fuzzy car controller using charts. The controller can be run using the `simulator.py` file. The fuzzy car controller utilizes fuzzy logic to make decisions based on the input parameters and control the car's behavior in a simulated environment.

## Project 3: Genetic Algorithm - Super Mario Game

In this project, a genetic algorithm is implemented to play the Super Mario game. The algorithm utilizes various genetic operators such as crossover, mutation, and selection to evolve a "goal chromosome" that represents an optimal strategy to complete the game levels. The goal is to find the best chromosome that achieves the highest score in the game.

To run the Super Mario game, the following steps are performed:
1. Select a game level.
2. Set the population size, scoring method, selection method, crossover method, crossover point, and mutation rate.
3. The genetic algorithm is applied to generate successive generations of chromosomes.
4. The algorithm terminates either when a maximum generation limit is reached or when a winning strategy is found.

Please refer to the individual project directories for more detailed information, source code, and instructions on running each project.

## Project Directory Structure

```
- Project 1 - Neural Network Implementation/
    - Source code files
    - Dataset files
    
- Project 2 - Fuzzy Car Controller/
    - Source code files
    - Simulator.py
    
- Project 3 - Genetic Algorithm - Super Mario Game/
    - Source code files
    - Game levels
```

Feel free to explore each project and delve into the details of their implementations.

**Note:** Make sure to follow the project-specific instructions provided in each project directory to set up the required dependencies and execute the code successfully.
