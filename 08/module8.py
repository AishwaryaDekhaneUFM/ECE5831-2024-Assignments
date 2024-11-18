import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from le_net import LeNet  # Assuming LeNet class is in lenet.py

def preprocess_image(image_filename):
    # Read the image using OpenCV in grayscale
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not open or find the image {image_filename}")
        sys.exit(1)
    
    # Resize image to 28x28 as expected by the model
    image = cv2.resize(image, (28, 28))
    
    # Normalize pixel values (invert the color so digits are white on black)
    image = 1 - (image / 255.0)
    
    # Expand dimensions to match model input shape (1, 28, 28, 1)
    image = np.expand_dims(image, axis=(0, -1))
    
    return image

def load_and_predict(model_path, img_path, true_digit):
    # Load the trained model
    model = LeNet()
    model.load(model_path)  # Load the saved model

    # Preprocess the input image
    img_array = preprocess_image(img_path)

    # Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = prediction[0]  # Get the predicted digit

    # Show the input image using matplotlib
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.title(f"Predicted: {predicted_digit}")
    # plt.show()

    # Check the prediction and print the result
    if predicted_digit == true_digit:
        print(f"Success: Image {img_path} is for digit {true_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {img_path} is for digit {true_digit} but the inference result is {predicted_digit}.")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="LeNet Digit Recognition")
    parser.add_argument('image_filename', type=str, help="Image filename (e.g., 3_2.png)")
    parser.add_argument('true_digit', type=int, help="True digit of the image")
    args = parser.parse_args()

    # Call the function to load the model and make predictions
    load_and_predict('dekhane', args.image_filename, args.true_digit)
