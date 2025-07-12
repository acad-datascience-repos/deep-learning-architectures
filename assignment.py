import tensorflow as tf

def train_cnn_on_mnist():
  """
  Loads the MNIST dataset, builds, compiles, and trains a simple CNN model.

  This function demonstrates the complete deep learning workflow:
  1. Data loading and preprocessing
  2. Model architecture definition
  3. Model compilation with optimizer and loss
  4. Model training and evaluation

  Returns:
    A tuple of (trained_model, training_history).
  """
  # Task 1: Load the MNIST dataset
  # Hint: Use tf.keras.datasets.mnist.load_data()
  (x_train, y_train), (x_test, y_test) = (None, None), (None, None)
  # Your code here

  # Task 2: Preprocess the data
  # Normalize pixel values to be between 0 and 1.
  # Reshape the images to (28, 28, 1).
  # Hint: Divide by 255.0 to normalize, use reshape(-1, 28, 28, 1) for channel dimension
  # Your code here

  # Task 3: Define the CNN model
  # Use a Sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
  # Hint: Use tf.keras.models.Sequential() with layers:
  # - Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
  # - MaxPooling2D((2, 2))
  # - Flatten()
  # - Dense(10, activation='softmax')
  model = None
  # Your code here

  # Task 4: Compile the model
  # Use the 'adam' optimizer and 'sparse_categorical_crossentropy' loss.
  # Hint: Use model.compile() with optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
  # Your code here

  # Task 5: Train the model
  # Use model.fit() and train for 3 epochs.
  # Hint: Use model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=0)
  history = None
  # Your code here

  return model, history