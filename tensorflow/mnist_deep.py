"""
Deep MNIST for Experts
https://www.tensorflow.org/get_started/mnist/pros
"""


# Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
# sess = tf.InteractiveSession()
sess = tf.Session()

# placeholders and variables 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))





# Build a Multilayer Convolutional Network

# Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
  
# Convolution and Pooling
# in this example: 
# convolutions: a stride of one and are zero padded so that the output is the same size as the input
# pooling: max pooling over 2x2 blocks

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
                        
# First Convolutional Layer
# convolution, 
# followed by max pooling.
W_conv1 = weight_variable([5, 5, 1, 32])  # [5, 5, 1, 32] -> shape 
# 5,5: first two dimensions are the patch size
# 1: is the number of input channels
# 32: is the number of output channels. 
b_conv1 = bias_variable([32]) # 32 -> shape 


# reshape X to a 4d tensor, with the second and third dimensions corresponding to image width and height
x_image = tf.reshape(x, [-1,28,28,1])  # 28*28
# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # will reduce the image size to 14x14.

# Second Convolutional Layer
# In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # reduced to 7x7

# Densely Connected Layer
# add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# To reduce overfitting, we will apply dropout before the readout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())  # Before Variables can be used within a session, they must be initialized using that session.


for i in range(1700):  # original 20000
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})  # add session=sess
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # add session=sess

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  # add session=sess