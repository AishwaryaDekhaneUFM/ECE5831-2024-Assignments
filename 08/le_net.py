from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class LeNet:
    def __init__(self, batch_size=32, epochs=10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()
    
    def _create_lenet(self):
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5,5), 
                   activation='sigmoid', input_shape=(28, 28, 1), 
                   padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            
            Conv2D(filters=16, kernel_size=(5,5), 
                   activation='sigmoid', 
                   padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        if self.model is None:
            print('Error: Create a model first..')
        
        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        

    def _preprocess(self):
        # load mnist data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # add channel dim
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

        # one-hot encoding
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                  batch_size=self.batch_size, 
                  epochs=self.epochs, 
                  validation_data=(self.x_test, self.y_test))
    
    def save(self, model_path_name):
        save_model(self.model, f"{model_path_name}_cnn_model.keras")
        print(f"Model saved as {model_path_name}_cnn_model.keras")

    def load(self, model_path_name):
        self.model = load_model(f"{model_path_name}_cnn_model.keras")
        print(f"Model loaded from {model_path_name}_cnn_model.keras")

    def predict(self, images):
        if self.model is None:
            print("Model is not loaded or trained.")
            return None
        
        # Preprocess images (normalize and add channel dimension)
        images = [img / 255.0 for img in images]  # normalize each image
        images = np.array(images).reshape(-1, 28, 28, 1)  # reshape with channel dimension
        
        predictions = self.model.predict(images)
        return predictions.argmax(axis=1)  # return the class with the highest probability

if __name__ == "__main__":
    # Training the model
    lenet = LeNet(batch_size=32, epochs=20)  # You can adjust batch size and epochs as needed
    lenet.train()

    # Saving the model
    lenet.save("dekhane")  # Saves as 'your_last_name_cnn_model.keras'