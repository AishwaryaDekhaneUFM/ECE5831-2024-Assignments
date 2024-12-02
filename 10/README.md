```markdown
# HW10 - Dogs vs. Cats Image Classification

## Overview
This assignment involves creating a neural network model to classify images of dogs and cats using the Dogs vs. Cats dataset from Kaggle. The task was to implement a Python class `DogsCats` to handle dataset preparation, model building, training, and prediction.

## Files
### Code Files
1. **dogs_cats.py**: Contains the implementation of the `DogsCats` class with methods for dataset preparation, model building, training, and prediction.
2. **module10.ipynb**: A Jupyter Notebook to demonstrate the functionality of the `DogsCats` class, from data preparation to training the model.

### Dataset
- **dogs-vs-cats-original**: Directory containing the original dataset downloaded from Kaggle.

## Classify Images of Dogs vs. Cats
### Implementation Details
- **Dataset Preparation**: The dataset is divided into training, validation, and test subsets.
  - Training: 2,400 - 11,999
  - Validation: 0 - 2,399
  - Test: 12,000 - 12,499
  
  The directory structure after splitting should look like this:
  ```
  dogs-vs-cats/
    ├── train/
    ├── valid/
    └── test/
  ```

- **Model Architecture**: A Convolutional Neural Network (CNN) with convolutional, pooling, and fully connected layers.

### Key Class Methods
- `__init__(self)`: Initializes `train_dataset`, `valid_dataset`, `test_dataset`, and `model` with `None`.
- `make_dataset_folders(self, subset_name, start_index, end_index)`: Creates different subsets of the dataset for training, validation, and testing.
- `_make_dataset(self, subset_name)`: Returns a `tf.data.Dataset` object for the specified subset.
- `make_dataset(self)`: Creates `train_dataset`, `valid_dataset`, and `test_dataset`.
- `build_network(self, augmentation=True)`: Builds and compiles the neural network model.
- `train(self, model_name)`: Trains the model using the `fit()` method and can use callbacks for better performance.
- `load_model(self, model_name)`: Loads a trained model from a file.
- `predict(self, image_file)`: Predicts the class of an image file.

### Usage
Perform the following steps in `module10.ipynb` to create and train the model:

1. **Import and Initialize**: Import the `DogsCats` class and initialize an object.
   ```python
   from dogs_cats import DogsCats
   dc = DogsCats()
   ```

2. **Create Dataset Folders**: Use `make_dataset_folders()` to create the train, validation, and test directories.
   ```python
   dc.make_dataset_folders('valid', 0, 2399)
   dc.make_dataset_folders('train', 2400, 11999)
   dc.make_dataset_folders('test', 12000, 12499)
   ```

3. **Create Dataset Objects**: Call `make_dataset()` to create dataset objects for training, validation, and testing.
   ```python
   dc.make_dataset()
   ```

4. **Build and Display the Model**: Build the neural network model and display its summary.
   ```python
   dc.build_network()
   dc.model.summary()
   ```

5. **Train the Model**: Train the neural network.
   ```python
   dc.train('model.first-name-last-name.identifier.keras')
   ```

### Results
After successfully training the model, you can load it and make predictions on new images using the `load_model()` and `predict()` methods, respectively.

## Model Link
The trained model can be found at the following link: [model link]

## Conclusion
This assignment provided practical experience in handling image datasets, building and training convolutional neural networks, and using TensorFlow for deep learning tasks. The `DogsCats` class provides a structured approach to manage the entire machine learning workflow from data preparation to model deployment.

**Prepared by:** [Aishwarya Dekhane]
```