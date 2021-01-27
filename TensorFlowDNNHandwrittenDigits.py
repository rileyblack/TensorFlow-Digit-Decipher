# ----------------------------------------------------------------------------------------------------------------------
# NOTE: excessive in-line comments were just for my own learning purposes
# ----------------------------------------------------------------------------------------------------------------------

# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm                                                                       # For progress bar
import matplotlib.pyplot as plt                                                             # For plotting
from sklearn.model_selection import train_test_split                                        # For easy data splitting


# Defining function to convert data into floats within [0, 1]
def normalize(x_denormalized, y_denormalized):
    x_denormalized = x_denormalized / np.float32(255)                                       # Converting data to floats within [0, 1]
    x_normalized = np.reshape(x_denormalized, (x_denormalized.shape[0], np.prod(x_denormalized.shape[1:])))  # Flattening data from (60000, 28, 28) to (60000, 784)
    y_normalized = y_denormalized
    return x_normalized, y_normalized


# Defining function to train neural network over single epoch
@tf.function
def train(train_dataset):
    training_loss.reset_states()                                                            # Resetting training loss evaluation metric
    training_accuracy.reset_states()                                                        # Resetting training accuracy evaluation metric
    for x, y in train_dataset:                                                              # For each batch of samples in data set
        with tf.GradientTape() as grad:
            y_predicted = neural_network(x)                                                 # Computing predictions of current model
            loss = loss_function(y, y_predicted)                                            # Finding loss between predictions and actual labels
            gradients = grad.gradient(loss, neural_network.trainable_variables)             # Computing the gradient of the loss WRT each model parameter
        optimizer.apply_gradients(zip(gradients, neural_network.trainable_variables))       # Updating model parameters at learning rate according to direction of steepest descent
        training_loss(loss)                                                                 # Accumulating loss between predictions and actual labels
        training_accuracy(y, y_predicted)                                                   # Accumulating accuracy between predictions and actual labels


# Defining function to test neural network
@tf.function
def test(test_dataset):
    validation_loss.reset_states()                                                          # Resetting validation loss evaluation metric
    validation_accuracy.reset_states()                                                      # Resetting validation accuracy evaluation metric
    for x, y in test_dataset:                                                               # For each batch of samples in data set
        y_predicted = neural_network(x)                                                     # Computing predictions of final model
        loss = loss_function(y, y_predicted)                                                # Finding loss between predictions and actual labels
        validation_loss(loss)                                                               # Accumulating loss between predictions and actual labels
        validation_accuracy(y, y_predicted)                                                 # Accumulating accuracy between predictions and actual labels


# Defining neural network class
class Net(keras.Model):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.dense1 = keras.layers.Dense(64, input_shape=[input_size], activation='relu')   # Defining first linear layer with 784 (24x24) inputs, 64 outputs, and ReLU activation function
        self.dense2 = keras.layers.Dense(output_size, input_shape=[64], activation='softmax')  # Defining second linear layer with 64 inputs, 10 outputs, and softmax activation function

    def call(self, x):                                                                      # Defining data flow through neural network
        layer_1 = self.dense1(x)                                                            # Passing data through layer 1 then capping with RELU function
        layer_2 = self.dense2(layer_1)                                                      # Passing data through layer 2 then capping with softmax function
        return layer_2


# Defining training constants
EPOCH_NUMBER = 5                                                                            # Number of passes over entire training dataset
BATCH_SIZE = 32                                                                             # Number of samples loaded per batch (for performance enhancement)

# Importing 28x28 grayscale images from 10 classes of hand written digits (60k for training, 10k for validation/testing)
(x_training, y_training), (x_testing, y_testing) = mnist.load_data()

# Evenly splitting testing data into validation set and test set (validation set used to intermediately evaluate "test" performance throughout learning)
x_validation, x_testing, y_validation, y_testing = train_test_split(x_testing, y_testing, test_size=0.5, random_state=1)

# Normalizing data into floats within [0, 1]
x_training, y_training = normalize(x_training, y_training)
x_validation, y_validation = normalize(x_validation, y_validation)
x_testing, y_testing = normalize(x_testing, y_testing)

# Creating dataset (batching is for processing efficiency, shuffling is to avoid recurring local minima)
training_dataset = tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(60000).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).shuffle(5000).batch(BATCH_SIZE)
testing_dataset = tf.data.Dataset.from_tensor_slices((x_testing, y_testing)).shuffle(5000).batch(BATCH_SIZE)

# Extracting key data set characteristics
image_pixels = x_training.shape[1]
number_of_labels = int(y_training.max() - y_training.min() + 1)

# Creating neural network
neural_network = Net(image_pixels, number_of_labels)                                        # Configuring model with 784 image pixels as input and and 10 label classes as output

# Defining loss (error) function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()                             # Quantifying difference between predicted label and actual label (needs normalized softmax y_pred input)

# Defining parameter updating algorithm
optimizer = tf.keras.optimizers.Adam()                                                      # Adam is alternative to classic gradient descent

# Defining training/validation evaluation metrics
training_loss = tf.keras.metrics.Mean(name='training_loss')
training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

# Defining empty containers to track evaluation metrics over training
loss_history = []
accuracy_history = []

# Training the neural network
for epoch in tqdm(range(EPOCH_NUMBER)):                                                     # For each pass over the full data set (tqdm for runtime progression bar display)
    train(training_dataset)                                                                 # Training the neural network
    test(validation_dataset)                                                                # Testing the neural network
    loss_history.append((training_loss.result(), validation_loss.result()))                 # Tracking training/validation loss over current epoch
    accuracy_history.append((training_accuracy.result(), validation_accuracy.result()))     # Tracking training/validation accuracy over current epoch

# Plotting training/validation losses
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_history)), np.array(loss_history)[:, 0])                            # Plotting training loss
plt.plot(range(len(loss_history)), np.array(loss_history)[:, 1])                            # Plotting validation loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])

# Plotting training/validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(len(loss_history)), np.array(accuracy_history)[:, 0])                        # Plotting training accuracy
plt.plot(range(len(loss_history)), np.array(accuracy_history)[:, 1])                        # Plotting validation accuracy
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# Testing test data
test(testing_dataset)

# Printing testing results
print(f"\n\nTesting Loss: {validation_loss.result()}\nTesting Accuracy: {validation_accuracy.result()}")
