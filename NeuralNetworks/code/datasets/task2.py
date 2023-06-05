import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from NeuralNetworks.code.layers.fullyconnected import FC
from NeuralNetworks.code.layers.convolution2d import Conv2D
from NeuralNetworks.code.layers.maxpooling2d import MaxPool2D
from NeuralNetworks.code.activations.activations import get_activation
from NeuralNetworks.code.losses.meansquarederror import MeanSquaredError
from NeuralNetworks.code.losses.binarycrossentropy import BinaryCrossEntropy
from NeuralNetworks.code.models.model import Model
from NeuralNetworks.code.optimizers.adam import Adam
from NeuralNetworks.code.optimizers.gradientdescent import GD

df = pd.read_csv("mnist_dataset.csv")
num_rows, num_cols = df.shape

# Separate features (pixels) and labels
X = df.drop('label', axis=1)  # Features (pixel columns)
y = df['label']  # Labels

scaler = MinMaxScaler()
# Fit the scaler to your data
scaler.fit(X)
# Apply the normalization to your data
X_normalized = scaler.transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
arch = {
    'Conv1': Conv2D(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv1'),
    'MaxPool1': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'Conv2': Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv2'),
    'MaxPool2': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'Conv3': Conv2D(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv3'),
    'MaxPool3': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'Conv4': Conv2D(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), name='Conv4'),
    'MaxPool4': MaxPool2D(kernel_size=(2, 2), stride=(2, 2), mode='max'),
    'Activation1': get_activation('relu'),
    'FC1': FC(input_size=256, output_size=1, name='FC1'),
    'Activation2': get_activation('sigmoid'),  # Use sigmoid activation for binary classification
}

criterion = BinaryCrossEntropy()
optimizer = Adam(arch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

model = Model(arch, criterion, optimizer)

epochs = 10
batch_size = 32

train_loss, val_loss = model.train(X_train, y_train, epochs, val=None, batch_size=batch_size, shuffling=True, verbose=1)


# Save the trained model
model.save('trained_model')

# Load the saved model
saved_model = Model.load('trained_model')

# predict
y_pred = saved_model.predict(X_test)

# Evaluate the performance on the test set
test_loss = criterion.loss(y_pred, y_test)
print('Test Loss:', test_loss)