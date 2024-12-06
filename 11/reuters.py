import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

class Reuters:
    def __init__(self, num_words=10000, num_classes=46):
        self.num_words = num_words
        self.num_classes = num_classes
        self.model = None

    def prepare_data(self):
        # Load Reuters dataset
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = reuters.load_data(num_words=self.num_words)
        
        # Vectorize input data
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)
        
        # One-hot encode labels
        self.y_train = to_categorical(self.train_labels, self.num_classes)
        self.y_test = to_categorical(self.test_labels, self.num_classes)
    
    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        # Build a Sequential model
        self.model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.num_words,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        # Compile the model
        self.model.compile(optimizer="rmsprop",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, epochs=20, batch_size=512, validation_split=0.2):
        # Train the model
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
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
        # Plot training and validation accuracy
        history_dict = self.history.history
        accuracy = history_dict['accuracy']
        val_accuracy = history_dict['val_accuracy']
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        return test_loss, test_acc
