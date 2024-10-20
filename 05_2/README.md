README for HW5-2 MnistData Class Implementation

Overview
This assignment involves the implementation of the `MnistData` class to handle the MNIST dataset, including downloading the data, loading images and labels, preprocessing (e.g., one-hot encoding), and managing the dataset for training and testing purposes. We also implemented a softmax function, ensuring it handles overflow errors appropriately. Finally, we tested the class functionality by visualizing samples from both the training and testing datasets.

Files Provided
- `module5-2.ipynb`: Jupyter Notebook showcasing the development process of the `MnistData` class along with various function implementations, visualizations, and examples.
- `mnist_data.py`: Python script defining the `MnistData` class.
- `module5-2.py`: Python script for command-line testing of the `MnistData` class using two input arguments: dataset type (train/test) and an index number for visualization.
- `readme.md`: Documentation of the assignment.

Structure of `mnist_data.py`
`MnistData` Class
This class manages the MNIST dataset by handling downloading, loading, preprocessing, and providing an interface to access training and testing data.

Methods:
1. _download: Method to download a single MNIST data file.
2. _download_all: Method to download all MNIST data files.
3. _load_images: Method to load images from the dataset file.
4. _load_labels: Method to load labels from the dataset file.
5. _create_dataset(): Method to structure the dataset for easy access.
6. _change_one_hot_label(): Method to convert labels to one-hot encoding.
7. _init_dataset: Method to initialize the dataset.
8. load: Method to return the train and test datasets.

Softmax Function
A numerically stable version of the softmax function to prevent overflow errors during computation.

Usage Example
```python
if __name__ == "__main__":
    print("MnistData class is to load MNIST datasets.")
    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()
    print(f"Train Images Shape: {train_images.shape}, Train Labels Shape: {train_labels.shape}")
    print(f"Test Images Shape: {test_images.shape}, Test Labels Shape: {test_labels.shape}")
```

 Structure of `module5-2.py`
 Command-Line Arguments
1. dataset_type: `train` or `test`.
2. index: Index of the image/label to visualize.

 Usage Example
```bash
python module5-2.py train 0
```

 Expected Output
- Displays the image corresponding to the index in the specified dataset (train/test).
- Prints the one-hot-encoded label and its corresponding digit.

 Steps of Assignment
1. Implement MnistData Class:
   - Create methods for downloading, loading, and preprocessing the MNIST dataset.
   - Ensure data integrity by visualizing samples.
2. Softmax Function:
   - Implement a numerically stable softmax function.
   - Test the function with example inputs to demonstrate stability.
3. Testing and Visualization:
   - Use the `module5-2.py` script to test the implementation from the command-line interface.
   - Visualize dataset samples to ensure proper loading and preprocessing.
4. Documentation:
   - Create a comprehensive `readme.md` detailing the implementation, usage, and testing instructions.

Sample Outputs of .py file-
https://drive.google.com/file/d/1xt6qtd8VT2qsPOGfll-a15F3w-GRlNjq/view?usp=sharing





