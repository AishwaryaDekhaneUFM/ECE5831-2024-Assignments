import os
import gzip
import pickle
import urllib
import numpy as np
import matplotlib.pyplot as plt

class MNIST_Data:
    # Define the directory where dataset is located
    dir_data = '/Users/aishwaryadekhane/Desktop/My_Files/Sem-3/PRNN/HW6/06/dataset'
    pkl_data = 'mnist.pkl'

    # Base URL to download the dataset from
    url = 'http://jrkwon.com/data/ece5831/mnist/'

    # Each image contains pixels (28x28)
    img_size = 28*28

    # File names of the MNIST dataset components
    key_file = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        self.dataset = {}
        self.pkl_dataset_path = os.path.join(self.dir_data, self.pkl_data)
        self.setup_dataset()

    def one_hot_encode(self, y, num_class):
        encoded = np.zeros((y.size, num_class))
        for idx, row in enumerate(encoded):
            row[y[idx]] = 1
        return encoded

    def _fetch(self, f_name):
        file_path = os.path.join(self.dir_data, f_name)
        if os.path.exists(file_path):
            print(f"File: {f_name} already exists locally.")
        else:
            print(f"Downloading {f_name} from the server...")
            urllib.request.urlretrieve(self.url + f_name, file_path)
            print('Download complete.')

    def download_all(self):
        for file_name in self.key_file.values():
            self._fetch(file_name)

    def load_labels(self, f_name):
        with gzip.open(f_name, 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)
        return labels

    def load_images(self, f_name):
        with gzip.open(f_name, 'rb') as file:
            images = np.frombuffer(file.read(), np.uint8, offset=16)
        return images.reshape(-1, self.img_size)

    def create_dataset(self):
        self.dataset['train_images'] = self.load_images(os.path.join(self.dir_data, self.key_file['train_images']))
        self.dataset['train_labels'] = self.load_labels(os.path.join(self.dir_data, self.key_file['train_labels']))
        self.dataset['test_images'] = self.load_images(os.path.join(self.dir_data, self.key_file['test_images']))
        self.dataset['test_labels'] = self.load_labels(os.path.join(self.dir_data, self.key_file['test_labels']))

        with open(self.pkl_dataset_path, 'wb') as pkl_file:
            print(f'Creating pickle file: {self.pkl_dataset_path}')
            pickle.dump(self.dataset, pkl_file)
            print('Pickle creation complete.')

    def setup_dataset(self):
        self.download_all()
        if os.path.exists(self.pkl_dataset_path):
            with open(self.pkl_dataset_path, 'rb') as pkl_file:
                print(f"Loading dataset from pickle: {self.pkl_dataset_path}")
                self.dataset = pickle.load(pkl_file)
                print('Dataset loaded successfully.')
        else:
            self.create_dataset()

    def load(self):
        # Scale image pixel values
        for key in ('train_images', 'test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32) / 255.0
        # Convert labels to one-hot encoding
        for key in ('train_labels', 'test_labels'):
            self.dataset[key] = self.one_hot_encode(self.dataset[key], 10)
        return (self.dataset['train_images'], self.dataset['train_labels']), (self.dataset['test_images'], self.dataset['test_labels'])


# When the script is executed, load and display part of the MNIST dataset
if __name__ == '__main__':
    mnist_data = MNIST_Data()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()
    print("MNIST_Data class for loading the MNIST dataset:")
    print("Calling load()")
    print("Returns (train_images, train_labels), (test_images, test_labels)")
    print("Images are flattened into 784-element arrays. To view, reshape the array.")
    print("Labels are one-hot encoded. Use np.argmax to get the numeric label.")

    number = 51
    image = test_images[number].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {np.argmax(test_labels[number])}")
    plt.show()
