# HW8: LeNet CNN Model for Digit Recognition

## Overview
This project implements a **LeNet Convolutional Neural Network (CNN)** model for recognizing handwritten digits (0-9) using the MNIST dataset. The model is trained and saved, and we load it to make predictions on custom digit images. The project includes scripts to train, save, and load the model, and predict a given input image while validating its accuracy.

## Project Structure
- **`le_net.py`**: Defines the `LeNet` class, which builds, compiles, trains, and saves the CNN model.
- **`module8.py`**: Loads the saved model and predicts the class of a provided image. Displays the image and shows whether the prediction is correct or not.
- **`batch.py`**: Runs `module8.py` on multiple images in batch mode for testing purposes.

## Prerequisites
- **Python 3.10+**
- **Packages**: Install the following packages if not already installed:
  ```bash
  pip install tensorflow opencv-python numpy argparse
  ```

## File Descriptions
### le_net.py
This file contains the `LeNet` class which:
- **Creates the LeNet CNN model** with layers including `Conv2D`, `AveragePooling2D`, and `Dense`.
- **Compiles the model** using `categorical_crossentropy` loss and `Adam` optimizer.
- **Trains the model** on the MNIST dataset.
- **Saves** the trained model to disk.
- **Loads** a saved model for predictions to avoid retraining each time.

### module8.py
This script uses a pretrained model to predict a single image of a handwritten digit and displays the image with the prediction result. It provides:
- **Image Display**: Shows the input image using `imshow`.
- **Prediction Verification**: Checks the prediction and provides feedback on whether it matched the expected digit.

#### Usage:
```bash
python module8.py <image_filename> <true_digit>
```
**Example:**
```bash
python module8.py 3_2.png 3
```
- If the prediction matches `true_digit`, it prints:
  ```
  Success: Image 3_2.png for digit 3 is recognized as 3.
  ```
- If the prediction is incorrect:
  ```
  Fail: Image 3_2.png for digit 3 is misclassified as <predicted_digit>.
  ```

### batch.py
This script runs `module8.py` on multiple images in a specified folder, helping automate testing across multiple samples.

#### Usage:
```bash
python batch.py
```
Ensure that paths to `module8.py` and test images are correctly set in this script.

## Model Training and Saving
To train and save the model:
1. Run the following commands in Python or Jupyter Notebook:
   ```python
   from le_net import LeNet
   model = LeNet(batch_size=32, epochs=20)
   model.train()
   model.save("dekhane")
   ```
   This will save the model as `dekhane_cnn_model.keras`.

## Model Loading and Prediction
To use the saved model without retraining:
1. In `module8.py`, ensure the model loads correctly:
   ```python
   model = LeNet()
   model.load("dekhane")  # Loads the trained model from "dekhane_cnn_model.keras"
   ```

## Important Notes
- **Image Preprocessing**: Each image is read in grayscale, resized to 28x28, normalized, and reshaped to fit the input shape of the CNN model.
- **File Path**: Ensure that model and image paths are correct in the scripts.
- **Error Handling**: When loading images, if an image cannot be found or opened, the script will print an error and exit.

## Example Workflow
1. Train the model and save it using `le_net.py`.
2. Run `module8.py` with custom images to predict and check accuracy.

---

**Assignment Completed by**: Aishwarya Dekhane

