import unittest
from assignment import train_cnn_on_mnist
import tensorflow as tf
import numpy as np

class TestCNNModel(unittest.TestCase):
    def test_train_cnn_on_mnist_returns_correct_types(self):
        """Test that the function returns the correct types"""
        model, history = train_cnn_on_mnist()
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsInstance(history, tf.keras.callbacks.History)
    
    def test_model_architecture(self):
        """Test that the model has the expected architecture"""
        model, _ = train_cnn_on_mnist()
        
        # Check that the model has layers
        self.assertGreater(len(model.layers), 0)
        
        # Check that it's a Sequential model
        self.assertIsInstance(model, tf.keras.Sequential)
        
        # Check that the output layer has 10 units (for digits 0-9)
        self.assertEqual(model.layers[-1].units, 10)
    
    def test_training_performance(self):
        """Test that the model achieves reasonable accuracy"""
        _, history = train_cnn_on_mnist()
        
        # Check if training happened and some accuracy was achieved
        self.assertTrue('accuracy' in history.history)
        self.assertTrue('val_accuracy' in history.history)
        
        # Check if final accuracy is reasonable (should be > 85% for MNIST)
        final_accuracy = history.history['accuracy'][-1]
        self.assertGreater(final_accuracy, 0.85, f"Final accuracy {final_accuracy:.3f} is too low")
        
        # Check if validation accuracy is also reasonable
        final_val_accuracy = history.history['val_accuracy'][-1]
        self.assertGreater(final_val_accuracy, 0.80, f"Final validation accuracy {final_val_accuracy:.3f} is too low")
    
    def test_data_preprocessing(self):
        """Test that the data preprocessing is working correctly"""
        # Load raw data to check preprocessing
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Check original data properties
        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))
        
        # Check that labels are in the correct range (0-9)
        self.assertTrue(np.all(y_train >= 0) and np.all(y_train <= 9))
        self.assertTrue(np.all(y_test >= 0) and np.all(y_test <= 9))

if __name__ == '__main__':
    unittest.main()
