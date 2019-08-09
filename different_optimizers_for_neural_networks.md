# [Overview of different Optimizers for neural networks](https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3)

## objective of Machine Learning algorithm  
- goal: reduce the difference between the predicted output and the actual output  
    + Cost function(C) or Loss function (Cost functions are convex functions)  
- goal: minimize the cost function by finding the optimized value for weights  
- ensure that the algorithm generalizes well  
- gradient descent: to achieve this we run multiple iterations with different weights --> this helps to find the minimum cost  


## gradient descent  
- an iterative machine learning **optimization algorithm** to reduce the cost function  
- help models to make accurate predictions  
- Gradient indicates the direction of increase  
    + go in the opposite direction of the gradient -->  to find the minimum point in the valley  
    + update parameters in the negative gradient direction to minimize the loss  
    + `θ` is the weight parameter, `η` is the learning rate and `∇J(θ;x,y)` is the gradient of weight parameter `θ`  
    ![](https://miro.medium.com/max/414/1*6a9Gx2UlB1ksh92TabyGPQ.png)  



## types of gradient descent
1. Batch Gradient Descent (Vanilla Gradient Descent)  
2. Stochastic Gradient Descent  
3. Mini batch Gradient Descent  


## role of an optimizer  
- update the weight parameters --> to minimize the loss function  
- Loss function: guides --> telling optimizer if it is moving in the right direction to reach the global minimum  

## types of optimizers  
#### momentum  

![](https://miro.medium.com/max/497/1*dDB34j5iKVMSzjj6hEoGEw.png)  
    
- like a ball rolling downhill --> ball gain momentum (to move faster) as it rolls down the hill  
- accelerate Gradient Descent(GD) when we have surfaces that curve more steeply in one direction than in another direction  
- dampens the oscillation
- updating the weights: takes the gradient of the current step as well as the gradient of the previous time steps --> helps us move faster towards convergence  
- convergence happens faster when apply momentum optimizer to surfaces with curves  

#### nesterov accelerated gradient (NAG)  

![](https://miro.medium.com/max/700/1*M4tDfNcMF5GGb8QBKVdzNA.png)  
![](https://miro.medium.com/max/700/1*8_y56VUb0gMWeblSe1oUQw.png)  
- ike a ball rolling down the hill but knows exactly when to slow down before the gradient of the hill increases again  
- going down the hill where we can look ahead in the future --> can optimize our descent faster
- works slightly better than standard Momentum  

**We need to tune the learning rate in Momentum and NAG which is an expensive process.**

#### Adagrad — Adaptive Gradient Algorithm  

![](https://miro.medium.com/max/700/1*SqryO8o7BP0f-LeU_5C8zw.png)  
- Adagrad is an adaptive learning rate method  
    + adopt the learning rate to the parameters  
    + perform larger updates for infrequent parameters and smaller updates for frequent parameters  
    - well suited when we have sparse data as in large scale neural networks  
    - e.g., GloVe word embedding uses adagrad where infrequent words required a greater update and frequent words require smaller updates  
- For SGD, Momentum, and NAG we update for all parameters `θ` at once. We also use the same learning rate `η`. In Adagrad we use different learning rate for every parameter `θ` for every time step `t`  
aaaaaaaa- Adagrad eliminates the need to manually tune the learning rate  
    - In the denominator, we accumulate the sum of the square of the past gradients. Each term is a positive term so it keeps on growing to make the learning rate η infinitesimally small to the point that algorithm is no longer able learning  

**Adadelta, RMSProp, and adam tries to resolve Adagrad’s radically diminishing learning rates.**

#### Adadelta  

![](https://miro.medium.com/max/457/1*XGNbWQnYMgVY5Yywna7mFQ.png)  
- an extension of Adagrad --> tries to reduce Adagrad’s aggressive, monotonically reducing the learning rate  
- does this by restricting the window of the past accumulated gradient to some fixed size of `w`
    + Running average at time `t` then depends on the previous average and the current gradient  
- do not need to set the default learning rate --> take the ratio of the running average of the previous time steps to the current gradient  


#### RMSProp  

![](https://miro.medium.com/max/700/1*adEDAdjulZUJisfzurVuWw.png)
- Root Mean Square Propagation (by Geoffrey Hinton)..
- tries to resolve Adagrad’s radically diminishing learning rates  
    + using a moving average of the squared gradient
    + utilizes the magnitude of the recent gradient descents to normalize the gradient  
- learning rate gets adjusted automatically and it chooses a different learning rate for each parameter  
- divides the learning rate by the average of the exponential decay of squared gradients  

    
#### Adam (Adaptive Moment Estimation)  

![](https://miro.medium.com/max/1050/1*_b-BaY8lOktUoLFb3YsBcQ.png)  
![](https://miro.medium.com/max/1050/1*g_NMoFaQxFS2r-lkTP84Vw.png)  
![](https://miro.medium.com/max/690/1*1TQNwqqotS7vMuPX1_Emcg.png)  
- one of the most popular gradient descent optimization algorithms  
    + computationally efficient + has very little memory requirement  
- calculates the individual adaptive learning rate for each parameter from estimates of first and second moments of the gradients  
- reduces the radically diminishing learning rates of Adagrad  
- a combination of **Adagrad** (works well on sparse gradients) and **RMSprop** (works well in online and nonstationary settings)  
- implements the exponential moving average of the gradients to scale the learning rate (instead of a simple average as in Adagrad)    + keeps an exponentially decaying average of past gradients  


#### Nadam (Nesterov-accelerated Adaptive Moment Estimation)  
- combines NAG and Adam
- employed for noisy gradients or for gradients with high curvatures
- learning process is accelerated by summing up the exponential decay of the moving averages for the previous and current gradient