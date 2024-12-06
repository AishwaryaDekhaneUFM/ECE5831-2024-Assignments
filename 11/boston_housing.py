import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models, layers

class BostonHousing:
    def __init__(self):
        self.model = None

    def prepare_data(self):
        # Load the Boston Housing dataset
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = boston_housing.load_data()

        # Normalize the data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data = (self.train_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std

    def build_model(self):
        # Build the regression model
        self.model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.train_data.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)  # Output layer for regression
        ])
        # Compile the model
        self.model.compile(optimizer="rmsprop",
                           loss="mse",
                           metrics=["mae"])

    def train(self, epochs=100, batch_size=16, validation_split=0.2):
        # Train the model
        self.history = self.model.fit(
            self.train_data,
            self.train_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def plot_loss(self):
        # Plot training and validation loss
        history_dict = self.history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        # Plot training and validation MAE
        history_dict = self.history.history
        mae = history_dict['mae']
        val_mae = history_dict['val_mae']
        epochs = range(1, len(mae) + 1)
        plt.plot(epochs, mae, 'bo', label='Training MAE')
        plt.plot(epochs, val_mae, 'b', label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.show()

    def evaluate(self):
        # Evaluate the model
        test_loss, test_mae = self.model.evaluate(self.test_data, self.test_targets)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
        return test_loss, test_mae
