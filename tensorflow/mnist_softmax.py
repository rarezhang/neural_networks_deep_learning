# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

### softmax
- assign probabilities to an object being one of several different things
- gives us a list of values between 0 and 1 that add up to 1

### softmax regression 
- add up the evidence of our input being in certain classes, 
- convert that evidence into probabilities.

### evidence of a given image 
- a weighted sum of the pixel intensities
- weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.
- extra evidence: bias
- the evidence for a class i given an input x is: evidence(i) = sum(Wi,j*Xj)+bi
- Wi is the weights and bi is the bias for class i, and j is an index for summing over the pixels in our input image x.
- convert the evidence tallies into our predicted probabilities y using the "softmax" function: y = softmax(evidence)
- softmax: exponentiating its inputs and then normalizing them. serving as "activation" or "link" function: softmax(x)i=exp(xi) / sum(exp(xj))
- one more unit of evidence increases the weight given to any hypothesis multiplicatively. And conversely, having one less unit of evidence means that a hypothesis gets a fraction of its earlier weight. Softmax normalizes weights, so that they add up to one, forming a valid probability distribution. 




"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  # A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]

  # Create the model
  # input: 
  x = tf.placeholder(tf.float32, [None, 784])  # 28*28=784-> flattened into a 784-dimensional vector
  # parameters: 
  #  A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
  #  can be used and even modified by the computation
  W = tf.Variable(tf.zeros([784, 10]))  # we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors 
  b = tf.Variable(tf.zeros([10]))
  # model: 
  # multiply x by W with the expression tf.matmul(x, W)
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])  # a new placeholder to input the correct answers

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  # tf.log computes the logarithm of each element of y. 
  # we multiply each element of y_ with the corresponding element of tf.log(y). 
  # tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. 
  # tf.reduce_mean computes the mean over all the examples in the batch.
  # 
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  
  # difference: softmax_cross_entropy_with_logits on tf.matmul(x, W) + b)
  # Gradient descent: shifts each variable a little bit in the direction that reduces the cost.
  # add new operations to your graph which implement backpropagation and gradient descent
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
      
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  

  sess = tf.InteractiveSession() # launch the model in an InteractiveSession
  tf.global_variables_initializer().run()  # initialize the variables we created
  
  
  # Train
  # Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent. Ideally, we'd like to use all our data for every step of training because that would give us a better sense of what we should be doing, but that's expensive. So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.
  for _ in range(1000):  #  run the training step 1000 times
    # Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # We run train_step feeding in the batches data to replace the placeholders.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  # tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  # 0.9192
                                    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
