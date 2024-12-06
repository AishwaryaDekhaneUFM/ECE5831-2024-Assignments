# HW11 - Binary, Multiclass Classification, and Regression

## Overview

This assignment focuses on practicing binary classification, multiclass classification, and regression using deep learning techniques described in Chapter 4 of the textbook *Deep Learning with Python [2nd Edition]* by Francois Chollet.

## Requirements

### 1. Binary Classification - IMDB

- **Script**: `imdb.py`
- **Class**: `Imdb`
- **Member Functions**:
  - `prepare_data()`: Prepares the IMDB dataset.
  - `build_model()`: Constructs the neural network model for binary classification.
  - `train()`: Trains the model using the prepared data.
  - `plot_loss()`: Plots the training and validation loss.
  - `plot_accuracy()`: Plots the training and validation accuracy.
  - `evaluate()`: Evaluates the model on the test dataset to show the loss and accuracy.

### 2. Multiclass Classification - Reuters

- **Script**: `reuters.py`
- **Class**: `Reuters`
- **Member Functions**:
  - `prepare_data()`: Prepares the Reuters dataset.
  - `build_model()`: Constructs the neural network model for multiclass classification.
  - `train()`: Trains the model using the prepared data.
  - `plot_loss()`: Plots the training and validation loss.
  - `plot_accuracy()`: Plots the training and validation accuracy.
  - `evaluate()`: Evaluates the model on the test dataset to show the loss and accuracy.

### 3. Regression - Boston Housing

- **Script**: `boston_housing.py`
- **Class**: `BostonHousing`
- **Member Functions**:
  - `prepare_data()`: Prepares the Boston Housing dataset.
  - `build_model()`: Constructs the neural network model for regression.
  - `train()`: Trains the model using the prepared data.
  - `plot_loss()`: Plots the training and validation loss.
  - `plot_accuracy()`: Plots the training and validation accuracy.
  - `evaluate()`: Evaluates the model on the test dataset to show the loss and accuracy.

### 4. Jupyter Notebook

- **Notebook**: `module11.ipynb`
- The notebook should include implementations and demonstrations of the following classes:
  - `Imdb`
  - `Reuters`
  - `BostonHousing`

## How to Run

1. **Binary Classification (IMDB)**:
   - Run the `imdb.py` script.
   - Instantiate the `Imdb` class and call its member functions in the following sequence:
     ```python
     imdb = Imdb()
     imdb.prepare_data()
     imdb.build_model()
     imdb.train()
     imdb.plot_loss()
     imdb.plot_accuracy()
     imdb.evaluate()
     ```

2. **Multiclass Classification (Reuters)**:
   - Run the `reuters.py` script.
   - Instantiate the `Reuters` class and call its member functions in the following sequence:
     ```python
     reuters = Reuters()
     reuters.prepare_data()
     reuters.build_model()
     reuters.train()
     reuters.plot_loss()
     reuters.plot_accuracy()
     reuters.evaluate()
     ```

3. **Regression (Boston Housing)**:
   - Run the `boston_housing.py` script.
   - Instantiate the `BostonHousing` class and call its member functions in the following sequence:
     ```python
     boston_housing = BostonHousing()
     boston_housing.prepare_data()
     boston_housing.build_model()
     boston_housing.train()
     boston_housing.plot_loss()
     boston_housing.plot_accuracy()
     boston_housing.evaluate()
     ```

4. **Jupyter Notebook**:
   - Open `module11.ipynb` and run the cells demonstrating the use of `Imdb`, `Reuters`, and `BostonHousing` classes.

---