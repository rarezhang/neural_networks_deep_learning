import tensorflow as tf


# [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)

### API
#- TensorFlow Core: lowest level. provide complete programming control. We recommend TensorFlow Core for machine learning researchers and others who require fine levels of control over their models  
#- other: built on top of tensorflow core. e.g., tf.contrib  

### tensor
#- central unit of data  
#- consists of a set of primitive values shaped into an array of any number of dimensions  
#- rank: number of dimensions  
"""
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
"""

### computational graph
#- a series of TensorFlow operations arranged into a graph of nodes  
#    + Each node takes zero or more tensors as inputs and produces a tensor as an output. 
#- building the computational graph  
#- running the computational graph 

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

# printing the nodes does not output the values 3.0 and 4.0 as you might expect

# when evaluated, would produce 3.0 and 4.0, respectively. To actually evaluate the nodes, we must run the computational graph within a session.

sess = tf.Session()
print(sess.run([node1, node2]))
# [3.0, 4.0]

# build more complicated computations by combining Tensor nodes with operations
node3 = tf.add(node1, node2)
print(node3)
# Tensor("Add:0", shape=(), dtype=float32)
print(sess.run(node3))
# 7.0


# A graph can be paramaterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# like a function or lambda
adder_node = a + b # provide a shortcut for tf.add(a,b)
print(sess.run(adder_node, {a:3, b:4.5}))
# 7.5
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))
# [3. 7.]
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4}))
# 21  (3+4)*3=21



#  Variables allow us to add trainable parameters to a graph. 
# They are constructed with a type and initial value:
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# variables are not initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program, you must explicitly call global_variables_initializer()
init = tf.global_variables_initializer()
sess.run(init)  # Until we call sess.run, the variables are uninitialized.

# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x:[1,2,3,4]}))
# [ 0.          0.30000001  0.60000002  0.90000004]

# To evaluate the model on training data, we need a y placeholder to provide the desired values
y = tf.placeholder(tf.float32)

# we need to write a loss function.
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 23.66

#  improve this manually by reassigning the values of W and b to the perfect values of -1 and 1. A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 0.0


### tf.train API  
# TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function
# simplest optimizer is gradient descent.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
# [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
# correct value: [-1, 1]



### tf.contrib.learn
# TensorFlow provides higher level abstractions for common patterns, structures, and functionality. 
# a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:
'''
running training loops
running evaluation loops
managing data sets
managing feeding
'''

# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
# {'loss': 9.7161115e-09, 'global_step': 1000}



### define a custom model that works with tf.contrib.learn
# tf.contrib.learn.Estimator

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss= loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x=np.array([1., 2., 3., 4.])
y=np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
# {'loss': 4.1715114e-11, 'global_step': 1000}
