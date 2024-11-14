import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

file = 'AMD_weights.pkl'
img = 28 * 28  
size = 100
no = 10
network = TwoLayerNetWithBackProp(input_size=img, hidden_size=size, output_size=no)

def load_model_weights():
    with open(file, 'rb') as f:
        print(f"Loading weights from {file}")
        network.params = pickle.load(f)
    network.update_layer()

def preprocess_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {filepath}")
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = 1 - (image.astype(np.float32) / 255.0)  # Normalize and invert
    return image.flatten().reshape(1, -1)  # Flatten for model input

def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.title("Processed Image for Prediction")
    plt.axis('off')
    plt.show()

def predict_digit(filepath, true_digit):
    processed_img = preprocess_image(filepath)
    predicted = np.argmax(network.predict(processed_img))

    if predicted == true_digit:
        print(f"Success: Predicted {predicted} for image at {filepath}, which matches true digit {true_digit}.")
    else:
        print(f"Error: Predicted {predicted} for image at {filepath}, expected {true_digit}.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a handwritten digit image.")
    parser.add_argument("filepath", type=str, help="Path to the image to classify")
    parser.add_argument("true_digit", type=int, help="True digit in the image")
    args = parser.parse_args()
    load_model_weights()
    predict_digit(args.filepath, args.true_digit)

if __name__ == "__main__":
    main()
