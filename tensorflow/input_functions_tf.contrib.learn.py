"""
Building Input Functions with tf.contrib.learn
https://www.tensorflow.org/get_started/input_fn


### input_fn: preprocess and feed data into models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

# (input_fn) encapsulate the logic for preprocessing and piping data into your models
# body: the specific logic for preprocessing data (e.g., scrubbing out bad examples or feature scaling)
# return: final feature and label data 
# feature: A dict containing key/value pairs that map feature column names to Tensors (or SparseTensors) containing the corresponding feature data.
# labels: A Tensor containing your label (target) values: the values your model aims to predict.
"""
def my_input_fn():

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels
"""    



# pandas dataframes or numpy arrays, need to convert it to Tensor
"""
feature_column_data = [1, 2.4, 0, 9.9, 3, 120]
feature_tensor = tf.constant(feature_column_data)
"""
# sparse matrix 
"""
sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])

[[0, 6, 0, 0, 0]
 [0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0.5]]
"""


# Passing input_fn Data to Your Model
'''
classifier.fit(input_fn=my_input_fn(training_set), steps=2000)
'''
# or
''' 
def my_input_function_training_set():
  return my_input_function(training_set)

classifier.fit(input_fn=my_input_fn_training_set, steps=2000)
'''



# Importing the Housing Data


tf.logging.set_verbosity(tf.logging.INFO) # set logging verbosity to INFO for more detailed log output


COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"] # column names for the data set
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"  # distinguish features from the label

# read the three CSVs
training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
                             
# Defining FeatureColumns and Creating the Regressor
# create their FeatureColumns using the tf.contrib.layers.real_valued_column() function
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
                  
# instantiate a DNNRegressor for the neural network regression model.
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10], # a hyperparameter specifying the number of nodes in each hidden layer (here, two hidden layers with 10 nodes each)
                                          model_dir="/tmp/boston_model")

# create an input function, which will accept a pandas Dataframe and return feature column and label values as Tensors                                          
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels
    
#  train the neural network regressor, run fit with the training_set passed to the input_fn
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

# see how the trained model performs against the test data set. Run evaluate, and this time pass the test_set to the input_fn:

ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
# Retrieve the loss from the ev results and print it to output:
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

# Making Predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print ("Predictions: {}".format(str(predictions)))