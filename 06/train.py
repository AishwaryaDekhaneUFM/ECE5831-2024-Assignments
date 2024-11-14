import pickle
import numpy as np
from mnist import Mnist
import matplotlib.pyplot as plt
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

# Constants
pkl_file = 'AMD_weights.pkl' 
ip = 28 * 28  
hidden_size = 100    
op = 10  
itr = 10000    
batch = 16     
rate = 0.01   

def load_data():
    mnist = Mnist()
    return mnist.load()
    
def initialize_network():
    return TwoLayerNetWithBackProp(input_size=ip, hidden_size=hidden_size, output_size=op)

def train_network(network, x_train, y_train, x_test, y_test):
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch, 1)
    train_losses = []
    train_accs = []
    test_accs = []

    for i in range(itr):
        batch_mask = np.random.choice(train_size, batch)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grads = network.gradient(x_batch, y_batch)

        update_params(network, grads)

        train_losses.append(network.loss(x_batch, y_batch))

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, y_train)
            train_accs.append(train_acc)
            test_acc = network.accuracy(x_test, y_test)
            test_accs.append(test_acc)
            print(f'Epoch {i // iter_per_epoch}: Train accuracy = {train_acc}, Test accuracy = {test_acc}')

    return train_losses, train_accs, test_accs

def update_params(network, grads):
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= rate * grads[key]

def save_weights(network, file_path):
    with open(file_path, 'wb') as f:
        print(f'Saving weights to {file_path}')
        pickle.dump(network.params, f)
        print('Save complete.')

def plot_accuracy(train_accs, test_accs):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_accs))
    plt.plot(x, train_accs, label='Train Accuracy', marker=markers['train'])
    plt.plot(x, test_accs, label='Test Accuracy', linestyle='--', marker=markers['test'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

network = initialize_network()
train_losses, train_accs, test_accs = train_network(network, x_train, y_train, x_test, y_test)
save_weights(network, pkl_file)
plot_accuracy(train_accs, test_accs)
network.params = None
