import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from le_net import LeNet  # Assuming LeNet is defined in a separate file

def load_trained_model(model_path):
    lenet = LeNet()
    try:
        lenet.load(model_path)  # Load the model
        return lenet.model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def prepare_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = 1 - (image.astype(np.float32) / 255.0)  # Invert and normalize
    return image.reshape(1, 28, 28, 1)
    
def predict_digit(model, image_path, actual_digit):
    input_image = prepare_image(image_path)
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.title("Processed Image")
    plt.axis('off')
    plt.show()
    prediction = np.argmax(model.predict(input_image))
    if prediction == actual_digit:
        print(f"Success: The image at {image_path} is digit {actual_digit}.")
    else:
        print(f"Failure: The image at {image_path} is digit {actual_digit}, but the model predicted {prediction}.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python digit_predictor.py <image_path> <expected_digit>")
        sys.exit(1)
    image_path = sys.argv[1]
    expected_digit = int(sys.argv[2])
    model = load_trained_model("dekhane")
    predict_digit(model, image_path, expected_digit)

if __name__ == "__main__":
    main()
