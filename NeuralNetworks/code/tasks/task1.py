import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..layers.convolution2d import Conv2D
from ..layers.maxpooling2d import MaxPool2D
from ..layers.fullyconnected import FC
from ..activations.activations import Activation, get_activation
from ..losses.binarycrossentropy import BinaryCrossEntropy
from ..losses.meansquarederror import MeanSquaredError
from ..models.model import Model
from ..optimizers.adam import Adam
from ..optimizers.gradientdescent import GD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
train_data = pd.read_csv('datasets/california_houses_price/california_housing_train.csv')
test_data = pd.read_csv('datasets/california_houses_price/california_housing_test.csv')

# Step 2: Explore the dataset
print(train_data.head())
print(test_data.head())

# Step 3: Split the features and labels
X_train = train_data.drop('median_house_value', axis=1)
y_train = train_data['median_house_value']

X_test = test_data.drop('median_house_value', axis=1)
y_test = test_data['median_house_value']

# Step 4: Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Verify the preprocessed data
print(X_train_scaled[:5])
print(X_test_scaled[:5])

# Step 6: Define the architecture
arch = {
    'conv1': Conv2D(in_channels=3, out_channels=16, name='conv1', kernel_size=3, stride=1),
    'pool1': MaxPool2D(),
    'fc1': FC(input_size=32 * 6 * 6, output_size=120, name='fc1'),
    'fc2': FC(input_size=120, output_size=84, name='fc2'),
    'fc3': FC(input_size=84, output_size=10, name='fc3'),
    'activation': get_activation('relu')
}

# Step 7: Create the criterion (loss) function
criterion = BinaryCrossEntropy()  # or MeanSquaredError()

# Step 8: Create the optimizer
optimizer = Adam(arch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Step 9: Create the model instance
model = Model(arch, criterion, optimizer)

# Step 10: Train the model
epochs = 10  # Set the number of epochs
batch_size = 32  # Set the batch size
model.train(X_train_scaled, y_train, epochs, val=None, batch_size=batch_size, shuffling=False, verbose=1)

# Step 11: Perform predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 12: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Step 13: Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
