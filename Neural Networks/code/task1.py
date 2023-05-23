import pandas as pd
from sklearn.preprocessing import StandardScaler
from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC
from activations import Activation, get_activation
from losses.binarycrossentropy import BinaryCrossEntropy
from losses.meansquareerror import MeanSquaredError
from model import Model

# Step 1: Load the dataset
train_data = pd.read_csv('C:/Users/Samin/Desktop/University/Term 7/Computational Intelligence/Projects/Neural Networks/code/datasets/california_houses_price/california_housing_train.csv')
test_data = pd.read_csv('C:/Users/Samin/Desktop/University/Term 7/Computational Intelligence/Projects/Neural Networks/code/datasets/california_houses_price/california_housing_test.csv')

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
    'conv1': Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1),
    'pool1': MaxPool2D(kernel_size=2, stride=2),
    'conv2': Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1),
    'pool2': MaxPool2D(kernel_size=2, stride=2),
    'fc1': FC(in_features=32 * 6 * 6, out_features=120),
    'fc2': FC(in_features=120, out_features=84),
    'fc3': FC(in_features=84, out_features=10),
    'activation': get_activation('relu')
}

# Step 7: Create the criterion (loss) function
criterion = BinaryCrossEntropy()  # or MeanSquaredError()

# Step 8: Create the model instance
model = Model(arch, criterion)

# Step 9: Train the model
epochs = 10  # Set the number of epochs
batch_size = 32  # Set the batch size
model.train(X_train_scaled, y_train, epochs, val=None, batch_size=batch_size, shuffling=False, verbose=1)
