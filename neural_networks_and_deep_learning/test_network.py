# loading in the MNIST data
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# set up a Network with 3030 hidden neurons
from Network import Network
net = Network([784, 10, 10])

# use stochastic gradient descent to learn from the MNIST training_data over 30 epochs, 
# with a mini-batch size of 10, 
# and a learning rate of eta =3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)