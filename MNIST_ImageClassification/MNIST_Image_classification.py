import tensorflow as tf

# Define the callback class which stops training if accuracy greater than 0.99. Invoke on every epoch end
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("Reached 99% accuracy so cancelling training!")
      self.model.stop_training = True



#Load the mnist data from keras datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Instantiate the Callback class
callbacks = myCallback()

#Normalize the input data
x_train = x_train/255.0
x_test = x_test/255.0

# Define the model layers
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define the optimizer and the loss function. Also define the metrics to be computed.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using the fit() function
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# Use test data to validate the trained model
model.evaluate(x_test, y_test)