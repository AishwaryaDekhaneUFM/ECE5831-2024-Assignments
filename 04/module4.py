#Author: Aishwarya Dekhane
#PRNN HW4

import numpy as np
from multilayer_perceptron import MultiLayerPerceptron 

# Create an instance of the MultiLayerPerceptron class
mlp = MultiLayerPerceptron()

# Initialize the network
mlp.init_network()

# Define various input vectors and demonstrate the forward pass
input_data_1 = np.array([10.0, 5.0])
input_data_2 = np.array([3.5, 1.5])
input_data_3 = np.array([-2.0, 0.8])
input_data_4 = np.array([0.0, -1.2])
input_data_5 = np.array([1.0, 3.0])

# Perform a forward pass with different inputs
y1 = mlp.forward(input_data_1)
print("Output for input [7.0, 2.0]:", y1)

y2 = mlp.forward(input_data_2)
print("Output for input [3.5, 1.5]:", y2)

y3 = mlp.forward(input_data_3)
print("Output for input [-2.0, 0.8]:", y3)

y4 = mlp.forward(input_data_4)
print("Output for input [0.0, -1.2]:", y4)

y5 = mlp.forward(input_data_5)
print("Output for input [1.0, 3.0]:", y5)
