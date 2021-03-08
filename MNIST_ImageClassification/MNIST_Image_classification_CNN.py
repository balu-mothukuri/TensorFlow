"""MNIST Dataset - Train a model with 99.8% accuracy on Handwritten Digits

##Dataset Details
MNIST dataset has items of handwriting -- the digits 0 through 9. It has grey scale images, each of size 28X28 pixels. Each pixel in a grey scale image is of size 1 byte, so there are a total of 784 bytes.

##Goal of the Model
The goal is to train an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. to stop training once you reach that level of accuracy.

When it reaches 99.8% or greater it should print the following string to the console and stop training the model. "Reached 99.8% accuracy so cancelling training!".

##Solution
1. Use callbacks to stop training once the desired accuracy is reached
2. Use CNNs to improve the Image classification model.

"""

import tensorflow as tf

#Define the callback - Stop training if accuracy reaches 99.8%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.998):
      print('Reached 99.8% accuracy, so cancelling training')
      self.model.stop_training = True

# Load data from MNIST Keras API
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Instantiate the callback
callbacks = myCallback()

#Reshape the input to a 4D Tensor as the Convolution expects the input as (batch_size, height, width, depth)
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

#Normalize the input data
training_images = training_images/255.0
test_images = test_images/255.0


# Define the hidden layers of the model
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=256, activation='relu'),
                                    tf.keras.layers.Dense(units=128, activation='relu'),
                                    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Define the optimizer and loss function to be used by the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = 'accuracy')

#Print the model summary
model.summary()

# Train the model
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])

#Test the model
model.evaluate(test_images, test_labels)

