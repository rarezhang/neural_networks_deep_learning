"""
tf.contrib.learn Quickstart


tf.contrib.learn
- high-level machine learning API
- easy to configure, train, and evaluate a variety of machine learning models
"""

### Load CSVs containing Iris training/test data into a TensorFlow Dataset
import tensorflow as tf
import numpy as np
# load the training and test sets into Datasets 
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


### Construct a neural network classifier
# configure a Deep Neural Network Classifier 
# Specify that all features have real-value data
# defines the model's feature columns, which specify the data type for the features in the data set
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, 
                                            hidden_units=[10, 20, 10], # Three hidden layers, containing 10, 20, and 10 neurons, respectively. 
                                            n_classes=3, # Three target classes, representing the three Iris species.
                                            model_dir="/tmp/iris_model") # directory TensorFlow will save checkpoint data during model training

### Fit the model using the training data
# Pass as arguments your feature data (training_set.data), target values (training_set.target), and the number of steps to train (here, 2000):
# Fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# The state of the model is preserved in the classifier, which means you can train iteratively if you like. For example, the above is equivalent to the following:
# classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
# classifier.fit(x=training_set.data, y=training_set.target, steps=1000)

### Evaluate the accuracy of the model
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


### Classify new samples
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))