import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers

class IMDB:
    def __init__(self, num_words=10000):
        self.num_words = num_words
        self.model = None

    def prepare_data(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words=self.num_words)
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)
        self.y_train = np.asarray(self.train_labels).astype("float32")
        self.y_test = np.asarray(self.test_labels).astype("float32")
        
    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.num_words))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(16, activation="relu", input_shape=(self.num_words,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="rmsprop",
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

    def train(self, epochs=4, batch_size=512, validation_split=0.2):
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def plot_loss(self):
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
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        return test_loss, test_acc
