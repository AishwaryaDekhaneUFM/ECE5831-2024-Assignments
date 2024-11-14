# HW6: Two-Layer Neural Network with Backpropagation

## Overview
This assignment involves implementing a two-layer neural network with backpropagation, training it on the MNIST dataset, and testing the trained model with your own handwritten digits. The various components of the solution such as activation functions and loss functions will be organized into different Python files.

## Files and Directories
- `layers.py`
  - Implements the `Relu`, `Sigmoid`, `Affine`, and `SoftmaxWithLoss` classes.
  - Imports necessary packages and previously implemented classes (`Activations` and `Errors`).
  
- `activations.py`
  - Contains the `Activations` class.
  
- `errors.py`
  - Contains the `Errors` class.
  
- `two_layer_net_with_back_prop.py`
  - Contains the `TwoLayerNetWithBackProp` class.
  
- `train.py`
  - Script to train the MNIST model using the `TwoLayerNetWithBackProp` class.
  - Training hyperparameters:
    - Iterations: 10,000
    - Batch size: 16
    - Learning rate: 0.01
  - Saves the trained model as `your_last_name_mnist_model.pkl`.

- `module6.ipynb`
  - Jupyter notebook to validate your implementation, show training steps, accuracy graphs, and test results.
  
- `module6.py`
  - Script similar to `module5-3.py`, modified to use the trained model instead of a pre-trained one.
  - Test handwritten digit images from HW5-3.

## Completed Tasks

### 1. Implemented Classes in `layers.py`
- Implemented `Relu`, `Sigmoid`, `Affine`, and `SoftmaxWithLoss`.
- Imported necessary packages and used classes from `activations.py` and `errors.py`.

### 2. Built the Neural Network in `two_layer_net_with_back_prop.py`
- Implemented the `TwoLayerNetWithBackProp` class.

### 3. Trained the Model in `train.py`
- Implemented the MNIST training procedure using `TwoLayerNetWithBackProp`.
- Used specified hyperparameters.
- Saved the trained model as `your_last_name_mnist_model.pkl`.

### 4. Validated Implementation in `module6.ipynb`
- Documented steps to test and validate the implementations of the classes.
- Showed training steps, accuracy calculations on training and test datasets, and plotted accuracy graphs.
- Ensured `iter_per_epoch` is of integer data type to avoid issues during comparisons.

### 5. Tested the Model in `module6.py`
- Adapted `module5-3.py` to `module6.py` using the trained model.
- Tested the model with handwritten digit images.

### 6. Displayed Results
- Used `imshow` to show input images.
- Printed results indicating success or failure:
  - Failure: `Fail: Image 2_1.png is for digit 2 but the inference result is 3.`
  - Success: `Success: Image 2_1.png is for digit 2 is recognized as 2.`

### 7. Running the Python Scripts
- To run a Python script in the Jupyter Notebook:
  ```sh
  !python module6.ipynb
  ```

- To run `module6.py` with input arguments:
  ```sh
  $ python module6.py <image_filename> <digit>
  ```
  Example:
  ```sh
  $ python module6.py 3_2.png 3
  ```


