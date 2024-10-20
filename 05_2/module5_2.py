import argparse
import matplotlib.pyplot as plt
from mnist_data import MNIST_Data  # Importing custom MNIST data loader class

# Core function to run the script
def run():
    # Step A: Setting up command-line arguments
    arg_parser = argparse.ArgumentParser(description='Display an image from the MNIST dataset.')
    arg_parser.add_argument('data_type', choices=['train', 'test'], help="Choose between 'train' or 'test' dataset.")
    arg_parser.add_argument('img_index', type=int, help='Provide the index of the image to view.')
    params = arg_parser.parse_args()

    # Step B: Fetching the MNIST dataset
    data_loader = MNIST_Data()
    (train_imgs, train_lbls), (test_imgs, test_lbls) = data_loader.load()

    # Step C: Determine which dataset to use
    if params.data_type == 'train':
        selected_images = train_imgs
        selected_labels = train_lbls
    else:
        selected_images = test_imgs
        selected_labels = test_lbls

    # Step D: Access the image and label at the specified index
    selected_img = selected_images[params.img_index].reshape(28, 28)  # Reshape to 28x28 for plotting
    corresponding_label = selected_labels[params.img_index]

    # Step E: Display the image with the label
    plt.imshow(selected_img, cmap='gray')
    plt.title(f"Label: {corresponding_label}")
    plt.show()

    # Step F: Output the label in the terminal
    print(f"Label: {corresponding_label}")

# Ensure that the script is executed directly
if __name__ == "__main__":
    run()
