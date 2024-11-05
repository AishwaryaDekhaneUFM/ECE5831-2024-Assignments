# module5_3.py
import sys
import matplotlib.pyplot as plt
import mnist

def main():
    if len(sys.argv) != 3:
        print("Usage: python module5_3.py <image_filename> <digit>")
        sys.exit(1)
    
    image_filename = sys.argv[1]
    correct_digit = int(sys.argv[2])
    
    # Initialize Mnist model
    my_mnist = mnist.Mnist()
    my_mnist.init_network()

    # Predict the digit
    predicted_digit = my_mnist.predict_image(image_filename)
    
    # Load and display the image
    #image = plt.imread(image_filename)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    
    # Print the result
    if predicted_digit == correct_digit:
        print(f"Success: Image {image_filename} is for digit {correct_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {correct_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()