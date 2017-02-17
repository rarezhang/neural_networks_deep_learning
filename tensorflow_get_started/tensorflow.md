# [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)

### API
- TensorFlow Core: lowest level. provide complete programming control. We recommend TensorFlow Core for machine learning researchers and others who require fine levels of control over their models  
- other: built on top of tensorflow core. e.g., tf.contrib  

### tensor
- central unit of data  
- consists of a set of primitive values shaped into an array of any number of dimensions  
- rank: number of dimensions  
```
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

### computational graph
- a series of TensorFlow operations arranged into a graph of nodes  
- building the computational graph  
- running the computational graph  