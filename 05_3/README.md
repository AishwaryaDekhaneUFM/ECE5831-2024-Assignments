# MNIST Handwritten Digit Recognition

## Assignment Description
This assignment is based on Module 5 and involves recognizing handwritten digits using a neural network. You need to create and test a custom MNIST-like dataset using your own handwritten digits.

## Project Structure
```
.
├── mnist.py             # Contains the Mnist class definition
├── module5-3.py         # Script to test the handwritten digit recognition
├── module5-3.ipynb      # Jupyter Notebook for demonstrating the work and testing functions
├── README.md            # This README file
└── images/              # Directory containing your handwritten digit images
    ├── 0_0.png
    ├── 0_1.png
    ├── ...
    ├── 9_4.png
```

## Prerequisites
- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook
- CV2 (OpenCV)
- TensorFlow/Keras (if required)

## How to Execute

### 1. Training and Testing the MNIST Model
First, ensure your `mnist.py` script is complete and contains code for training and testing a model for MNIST digit recognition. Include your custom images in the folder named `images`.

### 2. Running the Test Script
You can test individual images using the `module5-3.py` script. Here's how you can do it:

In your terminal, run:
```sh
python module5-3.py <image_filename> <digit>
```
For example:
```sh
python module5-3.py images/3_2.png 3
```

This will display the input image and print whether the recognition succeeded or failed.

### 3. Running Tests in Jupyter Notebook
To document and demonstrate your testing in the Jupyter Notebook, including how you preprocessed images and tested functions, use `module5-3.ipynb`. Ensure it contains:

- Code cells showing the steps taken.
- Results of running the test images.

To execute the notebook:
```sh
jupyter notebook module5-3.ipynb
```

Once in the notebook, run the cells to see the results of your testing.

## Example Output
For a successful recognition:
```plaintext
Success: Image 3_2.png is for digit 3 and is recognized as 3.
```

For a failed recognition:
```plaintext
Fail: Image 3_2.png is for digit 3 but the inference result is 5.
```

## Conclusion
Ensure all components are working as expected by testing each of your handwritten images. Validate the correct functionality of your MNIST class and your custom digit images using `module5-3.py` and `module5-3.ipynb`.
