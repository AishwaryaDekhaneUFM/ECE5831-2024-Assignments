```markdown
# HW11: Binary, Multiclass Classification, and Regression

## Overview
This assignment implements binary classification, multiclass classification, and regression using Python and deep learning techniques based on Chapter 4 of *Deep Learning with Python (2nd Edition)* by François Chollet.

### Implemented Models
1. **Binary Classification**: Using the IMDB dataset
2. **Multiclass Classification**: Using the Reuters dataset
3. **Regression**: Using the Boston Housing dataset

The project contains Python classes and a Jupyter notebook showcasing the implementations.

---

## File Structure
- **`imdb.py`**: Contains the `Imdb` class for binary classification.
- **`reuters.py`**: Contains the `Reuters` class for multiclass classification.
- **`boston_housing.py`**: Contains the `BostonHousing` class for regression tasks.
- **`module11.ipynb`**: Jupyter notebook demonstrating:
  - Implementation of all three classes.
  - Training, evaluation, and visualization of results.

---

## Classes and Functions

### Binary Classification - IMDB (`imdb.py`)
**Class Name**: `Imdb`  
Member Functions:
- `prepare_data()`: Prepares and processes the IMDB dataset.
- `build_model()`: Builds the binary classification model.
- `train()`: Trains the model using the prepared data.
- `plot_loss()`: Plots training and validation loss.
- `plot_accuracy()`: Plots training and validation accuracy.
- `evaluate()`: Evaluates the model on the test dataset.

### Multiclass Classification - Reuters (`reuters.py`)
**Class Name**: `Reuters`  
Member Functions:
- `prepare_data()`: Prepares and processes the Reuters dataset.
- `build_model()`: Builds the multiclass classification model.
- `train()`: Trains the model using the prepared data.
- `plot_loss()`: Plots training and validation loss.
- `plot_accuracy()`: Plots training and validation accuracy.
- `evaluate()`: Evaluates the model on the test dataset.

### Regression - BostonHousing (`boston_housing.py`)
**Class Name**: `BostonHousing`  
Member Functions:
- `prepare_data()`: Prepares and processes the Boston Housing dataset.
- `build_model()`: Builds the regression model.
- `train()`: Trains the model using the prepared data.
- `plot_loss()`: Plots training and validation loss.
- `plot_accuracy()`: (Optional) Plots additional metrics, if applicable.
- `evaluate()`: Evaluates the model on the test dataset.

---

## Jupyter Notebook: `module11.ipynb`
This notebook includes:
1. Instantiating each class (`Imdb`, `Reuters`, and `BostonHousing`).
2. Demonstrating:
   - Data preparation
   - Model building
   - Training
   - Plotting (loss and accuracy/metrics)
   - Evaluation
3. Results of accuracy and performance metrics.

---

## Accuracy Threshold
To meet assignment requirements, ensure:
- **Binary Classification** and **Multiclass Classification** achieve an accuracy of **≥ 0.6**.
- **Regression** provides meaningful predictions.

---

## How to Run
1. Clone the repository or download the files.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook (`module11.ipynb`) to view the results:
   ```bash
   jupyter notebook module11.ipynb
   ```

---

## Dependencies
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Author
Aishwarya Dekhane
```
