"""
a module to implement the stochastic gradient descent learning algorithm for a feed forward neural network. 

Gradients are calculated using back-propagation.

Not the optimized version
"""

import random
import numpy as np

class Netwrok(object):
	
	def __init__(self, sizes):
		"""
		sizes: [python list] the number of neurons in the respective layers of the network
		e.g., [2,3,1] -> three layer network
		1st layer: 2 neurons -> the first layer is assumed to be an input layer
		2nd layer: 3 neurons
		3rd layer: 1 neurons
		
		bias / weights: initialized randomly (Gaussian distribution: mean 0, variance 1)
		"""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # list of numpy 1d array, each array has x elements(depends on the value in sizes)
		self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])] # e.g., [2,3,1] -> zip() -> [(2,3),(3,1)]
		# list of array [array(3x2),array(1x3))]
		
	def feedforword(self, a):
		"""
		return the output of the network if 'a' is input
		"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a
	
	def 