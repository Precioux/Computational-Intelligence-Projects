import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworks.code.layers.fullyconnected import FC
from NeuralNetworks.code.activations.activations import get_activation
from NeuralNetworks.code.losses.meansquarederror import MeanSquaredError
from NeuralNetworks.code.models.model import Model
from NeuralNetworks.code.optimizers.adam import Adam

# Step 1: Load the dataset
print('Step 1: Load the dataset')
data = pd.read_csv('california_houses_price/california_housing_train.csv')
data_test = pd.read_csv('california_houses_price/california_housing_test.csv')

# Step 2: Explore the dataset
print('Step 2: Explore the dataset')
print(data.head())
print('Number of samples: ')
print(data.shape[0])



# Step 3: Split the features and labels
print('Step 3: Split the features and labels')
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']


# Step 4: Normalize the features
print('Step 4: Normalize the features')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the data into train and test sets
print('Step 5: Split the data into train and test sets')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Define the architecture
print('Step 6: Define the architecture')
arch = {
    'FC1': FC(input_size=X_train.shape[1], output_size=128, name='FC1'),
    'activation1': get_activation('relu'),
    'FC2': FC(input_size=128, output_size=64, name='FC2'),
    'activation2': get_activation('relu'),
    'FC3': FC(input_size=64, output_size=32, name='FC3'),
    'activation3': get_activation('relu'),
    'FC4': FC(input_size=32, output_size=1, name='FC4'),
    'activation4': get_activation('linear'),  # Add linear activation for the final layer
}

# Step 7: Create the criterion (loss) function
print('Step 7: Create the criterion (loss) function')
criterion = MeanSquaredError()

# Step 8: Create the optimizer
print('Step 8: Create the optimizer')
optimizer = Adam(arch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Step 9: Create the model instance
print('Step 9: Create the model instance')
model = Model(arch, criterion, optimizer)

# Step 10: Train the model with validation
print('Step 10: Train the model with validation')
epochs = 10  # Set the number of epochs
batch_size = 3  # Set the batch size

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  # Create a validation set
train_loss, val_loss = model.train(X_train, y_train, epochs, val=(X_val, y_val), batch_size=batch_size, shuffling=True, verbose=1)

# Step 11: Perform predictions on the test set
print('Step 11: Perform predictions on the test set')
y_pred = model.predict(X_test)

# Step 12: Evaluate the model
print('Step 12: Evaluate the model')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)

# Step 13: Visualize the predictions
print('Step 13: Visualize the predictions')
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Step 14: Preprocess the data_test dataset
# Drop unnecessary columns and normalize the features
X_test_processed = scaler.transform(data_test.drop('median_house_value', axis=1))

# Step 15: Perform predictions on the data_test dataset
y_test_pred = model.predict(X_test_processed)

# Step 16: Evaluate the model on the data_test dataset
mse_test = mean_squared_error(data_test['median_house_value'], y_test_pred)
rmse_test = np.sqrt(mse_test)
print("Test Set Mean Squared Error:", mse_test)
print("Test Set Root Mean Squared Error:", rmse_test)
