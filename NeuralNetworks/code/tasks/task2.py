import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from NeuralNetworks.code.layers.convolution2d import *
from NeuralNetworks.code.activations.activations import get_activation, Activation
from NeuralNetworks.code.layers.fullyconnected import FC
from NeuralNetworks.code.layers.maxpooling2d import MaxPool2D
from NeuralNetworks.code.losses.binarycrossentropy import BinaryCrossEntropy
from NeuralNetworks.code.models.model import Model
from NeuralNetworks.code.optimizers.adam import Adam

# Step 1: Define the paths to the image folders
image_folder = "MNIST"
class_2_folder = image_folder + "/2"
class_5_folder = image_folder + "/5"

# Step 2: Create an ImageDataGenerator instance and specify the preprocessing steps
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Step 3: Load the images from the folders and apply preprocessing
image_size = (28, 28)  # Assuming MNIST images are 28x28 pixels
batch_size = 32

# Load images for class 2
class_2_data = datagen.flow_from_directory(
    class_2_folder,
    target_size=image_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

# Load images for class 5
class_5_data = datagen.flow_from_directory(
    class_5_folder,
    target_size=image_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

# Step 4: Combine the data from both classes
combined_data = tf.data.Dataset.zip((class_2_data, class_5_data))

# Step 5: Split the combined data into training and validation sets
train_data = combined_data.take(800)  # Adjust the number of training samples as needed
val_data = combined_data.skip(800)   # Adjust the number of validation samples as needed

# Step 6: Extract the images and labels from the combined data
train_images = []
train_labels = []
val_images = []
val_labels = []

for (image_2, label_2), (image_5, label_5) in train_data:
    train_images.append(image_2)
    train_images.append(image_5)
    train_labels.append(label_2)
    train_labels.append(label_5)

for (image_2, label_2), (image_5, label_5) in val_data:
    val_images.append(image_2)
    val_images.append(image_5)
    val_labels.append(label_2)
    val_labels.append(label_5)

# Convert the lists to TensorFlow tensors
train_images = tf.concat(train_images, axis=0)
train_labels = tf.concat(train_labels, axis=0)
val_images = tf.concat(val_images, axis=0)
val_labels = tf.concat(val_labels, axis=0)

# Step 7 : Prepare the data
# Convert the images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Reshape the input images if needed
# train_images = train_images.reshape(train_images.shape[0], -1)
# val_images = val_images.reshape(val_images.shape[0], -1)

# Assign labels (0 for class 2, 1 for class 5)
train_labels = np.where(train_labels == 2, 0, 1)
val_labels = np.where(val_labels == 2, 0, 1)

# Step 8 : arch :  LeNet-5
arch = {
    'conv1': Conv2D(input_channels=1, num_filters=6, kernel_size=5, stride=1, padding='valid'),
    'activation1': Activation('relu'),
    'pool1': MaxPool2D(pool_size=2, stride=2),
    'conv2': Conv2D(input_channels=6, num_filters=16, kernel_size=5, stride=1, padding='valid'),
    'activation2': Activation('relu'),
    'pool2': MaxPool2D(pool_size=2, stride=2),
    'fc1': FC(input_size=16*5*5, output_size=120),
    'activation3': Activation('relu'),
    'fc2': FC(input_size=120, output_size=84),
    'activation4': Activation('relu'),
    'fc3': FC(input_size=84, output_size=2)
}

# Step 9: Create the criterion (loss) function
criterion = BinaryCrossEntropy()

# Step 10: Create the optimizer
optimizer = Adam(arch, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Step 11: Create the model instance
model = Model(arch, criterion, optimizer)

# Step 12: Train the model
train_cost, val_cost = model.train(train_images, train_labels, epochs=10, val=(val_images, val_labels), batch_size=32, verbose=1)

# Step 13: Evaluate the model
predictions = model.predict(val_images)
accuracy = accuracy_score(val_labels, np.round(predictions))
print("Accuracy:", accuracy)

# Step 14: Save the trained model
model.save("mnist_model.pkl")
