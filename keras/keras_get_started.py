"""
keras getting started
# https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/getting-started/functional-api-guide/
"""

# sequential model 
from keras.models import Sequential
model = Sequential()

# stacking layers 
# add layers via the .add() method:
from keras.layers import Dense, Activation
model.add(Dense(output_dim=64, input_dim=100))  # Dense: regular fully connected NN layer.
# Activation: Applies an activation function to an output.
model.add(Activation("relu"))  # rectifier: is an activation function defined as f(x)=max(0,x)
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))  # normalized exponential function.  a generalization of the logistic function that "squashes" a K-dimensional vector z of arbitrary real values to a K-dimensional vector sigma(z) of real values in the range (0, 1) that add up to 1
"""
# passing a list of layer instances to the constructor
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
"""

# merge layer
"""
# Multiple Sequential instances can be merged into a single output via a Merge layer.
from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')
# merged = Merge([left_branch, right_branch], mode=lambda x: x[0] - x[1])  # pass a function as the mode argument

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='softmax'))
"""

# configure the learning process
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# further configure the optimizer
from keras.optimizer import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))


# iterate the training data in batches
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
"""
# feed batches to the model manually
model.train_on_batch(X_batch, Y_batch)
"""

# evaluate the performance 
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

# generate predictions on new data
classes = model.predict_classes(X_test, batch_size=32)
prob = model.predict_proba(X_test, batch_size=32)
