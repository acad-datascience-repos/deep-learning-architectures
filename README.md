# Deep Learning CNN Assignment

## Problem Description

In this assignment, you will build and train a simple Convolutional Neural Network (CNN) to classify images of handwritten digits from the famous MNIST dataset. This is a classic "Hello, World!" project for deep learning that introduces fundamental concepts of neural networks and computer vision.

## Learning Objectives

By completing this assignment, you will learn:
- How to load and preprocess image data for deep learning
- How to build a CNN architecture using TensorFlow/Keras
- How to compile and train a neural network model
- How to interpret training history and model performance
- Best practices for deep learning model development

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have the following packages installed:
   - tensorflow (>=2.10.0)
   - numpy (>=1.21.0)
   - matplotlib (>=3.5.0)

3. **Note**: This assignment requires significant computational resources. A GPU is recommended but not required.

## Instructions

1. Open the `assignment.py` file.
2. You will find a function definition: `train_cnn_on_mnist()`.
3. Your tasks are to:
   *   **Task 1**: Load the MNIST dataset using `tensorflow.keras.datasets.mnist.load_data()`.
   *   **Task 2**: Preprocess the image data: normalize the pixel values to be between 0 and 1, and reshape the images to include a channel dimension.
   *   **Task 3**: Define a Keras Sequential model with a `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` output layer.
   *   **Task 4**: Compile the model. Use the `'adam'` optimizer and `'sparse_categorical_crossentropy'` as the loss function.
   *   **Task 5**: Train the model on the training data for 3 epochs.
   *   The function should return the trained model and its training history.

## Hints

*   The MNIST dataset is included directly in TensorFlow/Keras.
*   Pixel values range from 0 to 255. You can normalize them by dividing by 255.0.
*   The original image shape is (28, 28). A CNN in Keras expects a 4D tensor, so reshape it to (28, 28, 1).
*   The final `Dense` layer should have 10 units (for digits 0-9) and a `softmax` activation.
*   The `model.fit()` function returns a `History` object that contains the training history.

## Testing Your Solution

Run the test file to verify your implementation:
```bash
python test.py
```

The tests will check:
- That your function returns the correct types (model and history)
- That the model has the expected architecture
- That the model achieves reasonable accuracy (>85%)
- That the data preprocessing is working correctly

## Expected Output

Your function should return a tuple containing:
- `model`: A trained TensorFlow/Keras CNN model
- `history`: A History object containing training metrics

The model should achieve at least 85% accuracy on the MNIST dataset after 3 epochs of training.

## Further Exploration (Optional)

*   How can you improve the model's accuracy? Try adding another `Conv2D` and `MaxPooling2D` layer. What happens?
*   Look up `Dropout` layers in Keras. How can adding a `Dropout` layer after the `Flatten` layer help prevent overfitting?
*   The `history` object can be used to plot the training and validation accuracy over time. How would you do this using matplotlib?
*   Try different optimizers (SGD, RMSprop) and compare their performance.
*   Experiment with different learning rates and see how they affect training.

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [MNIST Dataset Information](http://yann.lecun.com/exdb/mnist/)
- [CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
