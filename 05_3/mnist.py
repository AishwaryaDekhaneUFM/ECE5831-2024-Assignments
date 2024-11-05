import numpy as np
import pickle
import cv2
import mnist_data
import matplotlib.pyplot as plt

class Mnist:
    def __init__(self):
        self.data = mnist_data.MNIST_Data()
        self.params = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)
    
    def load(self):
        (x_train, y_train), (x_test, y_test) = self.data.load()
        return (x_train, y_train), (x_test, y_test)
    
    def init_network(self):
        with open('/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW5-3/model/sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)
    
    def display_image(self, image):
        """Display the processed image using matplotlib."""
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    # def preprocess_image(self, image_path):
    #     print('in the preprocessing')
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (28, 28))
    #     image = 1 - (image.astype('float32') / 255.0)
    #     image = image.flatten().reshape(1, -1)  # Flatten and add batch dimension
    #     return image

    def preprocess_image(self, image_path):
        """Load, preprocess, and display an image for prediction."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image {image_path} not found.")
        
        # Resize to 28x28 if not already, normalize, and flatten
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        image = 1 - (image.astype(np.float32) / 255.0)

        # Display the processed image
        self.display_image(image)
        
        return image.flatten().reshape(1, -1)  # Return flattened image for prediction
    
    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        return y

    def predict_image(self, image_path):
        image = self.preprocess_image(image_path)
        return np.argmax(self.predict(image), axis=1)
