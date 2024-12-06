

```markdown
# HW11 - Binary, Multiclass Classification, and Regression

## Overview
This assignment implements three machine learning tasks using deep learning principles:
1. **Binary Classification** - Sentiment analysis using the IMDB dataset.
2. **Multiclass Classification** - Topic classification using the Reuters dataset.
3. **Regression** - Predicting housing prices using the Boston Housing dataset.

All implementations are based on the textbook *Deep Learning with Python (2nd Edition)* by François Chollet.

## Structure
The project includes the following components:
- **`imdb.py`**: Implements the `Imdb` class for binary classification.
- **`reuters.py`**: Implements the `Reuters` class for multiclass classification.
- **`boston_housing.py`**: Implements the `BostonHousing` class for regression.
- **`module11.ipynb`**: A Jupyter Notebook demonstrating the implementation and results for all three classes.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib notebook
```

## Classes and Functions

### `Imdb` Class
Handles binary classification on the IMDB dataset (sentiment analysis).
- **`prepare_data()`**: Prepares and preprocesses the IMDB dataset.
- **`build_model()`**: Defines the architecture of the binary classification model.
- **`train()`**: Trains the model using the training dataset.
- **`plot_loss()`**: Plots the training and validation loss over epochs.
- **`plot_accuracy()`**: Plots the training and validation accuracy over epochs.
- **`evaluate()`**: Evaluates the model's performance on the test dataset.

### `Reuters` Class
Handles multiclass classification on the Reuters dataset (topic classification).
- **`prepare_data()`**: Prepares and preprocesses the Reuters dataset.
- **`build_model()`**: Defines the architecture of the multiclass classification model.
- **`train()`**: Trains the model using the training dataset.
- **`plot_loss()`**: Plots the training and validation loss over epochs.
- **`plot_accuracy()`**: Plots the training and validation accuracy over epochs.
- **`evaluate()`**: Evaluates the model's performance on the test dataset.

### `BostonHousing` Class
Handles regression on the Boston Housing dataset (price prediction).
- **`prepare_data()`**: Prepares and preprocesses the Boston Housing dataset.
- **`build_model()`**: Defines the architecture of the regression model.
- **`train()`**: Trains the model using the training dataset.
- **`plot_loss()`**: Plots the training and validation loss over epochs.
- **`plot_accuracy()`**: Plots the training and validation accuracy over epochs.
- **`evaluate()`**: Evaluates the model's performance on the test dataset.

### Jupyter Notebook
- **`module11.ipynb`**: Demonstrates the implementation of the `Imdb`, `Reuters`, and `BostonHousing` classes. Includes:
  - Dataset preparation.
  - Model architecture visualization.
  - Training progress.
  - Loss and accuracy plots.
  - Evaluation results.

## Usage
1. Run `module11.ipynb` in Jupyter Notebook to see the implementation and results.
2. Ensure the accuracy on test datasets exceeds **0.85** for classification tasks.

### Example Commands (in Notebook)
```python
# Binary Classification
from imdb import Imdb
imdb_model = Imdb()
imdb_model.prepare_data()
imdb_model.build_model()
imdb_model.train()
imdb_model.plot_loss()
imdb_model.plot_accuracy()
imdb_model.evaluate()

# Multiclass Classification
from reuters import Reuters
reuters_model = Reuters()
reuters_model.prepare_data()
reuters_model.build_model()
reuters_model.train()
reuters_model.plot_loss()
reuters_model.plot_accuracy()
reuters_model.evaluate()

# Regression
from boston_housing import BostonHousing
boston_model = BostonHousing()
boston_model.prepare_data()
boston_model.build_model()
boston_model.train()
boston_model.plot_loss()
boston_model.plot_accuracy()
boston_model.evaluate()
```
## Author
- **Aishwarya Dekhane**

## License
This project is for academic purposes and adheres to the University’s guidelines.
```
