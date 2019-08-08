# [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)  


## what is activation function  
- use to get the output of node  
- known as Transfer Functions  

## why use activation function  
- to determine the output of neural network  
    + e.g., yes or no  
    + e.g., maps the resulting values in between [0, 1] or [-1, 1] (depending upon the function)  
- normally __2__ types  
    + Linear Activation  
    + Non-linear Activation  
- Linear (or identity) Activation Functions  
    + the function is a **line** or **linear**  
    + the output of the functions will **not** be confined between any range  
    ![Fig: Linear Activation Function](https://miro.medium.com/max/700/1*tldIgyDQWqm-sMwP7m3Bww.png)  
    + Equation: $f(x)=x$  
    + Range: (-infinity, infinity)  
    + doesn’t help with the complexity or various parameters of usual data that is fed to the neural networks.  
- Non-linear activation functions  
    + the most used activation functions  
    ![Fig: Non-linear Activation Function](https://miro.medium.com/max/600/1*cxNqE_CMez7vUIkcLUH8PA.png)  
    + makes it easy for the model to generalize (or adapt) with variety of data and to differentiate between the output  
    + **Derivative or Differential**: Change in y-axis w.r.t. change in x-axis.It is also known as slope  
    + **Monotonic function**: A function which is either entirely non-increasing or non-decreasing  
    + mainly divided on the basis of their **range or curves**  
        1. Sigmoid (Logistic Activation)  
        ![Fig: Sigmoid Function](https://miro.medium.com/max/485/1*Xu7B5y9gp0iL5ooBj7LtWw.png)  
        - S shape  
        - **range** --> $(0, 1)$  
        - especially used for models where need to **predict the probability** as an puput (probability exists only between [0, 1])  
        - **differentiable** --> can find the slope of the sigmoid curve at any two points  
        - **monotonic** --> but function’s derivative is not  
        - can cause a neural network to get stuck at the training time  
        - **softmax function** is a more generalized logistic activation function which is used for **multiclass classification**  
        
        2. Tanh (Hyperbolic Tangent)  
        ![Fig: tanh v/s Logistic Sigmoid](https://miro.medium.com/max/595/1*f9erByySVjTjohfFdNkJYQ.jpeg)  
        - **range** --> $(-1, 1)$  
        - tanh function is mainly used classification between **two classes**  
        - sigmoidal --> S shape  
        - advantage: the **negative inputs** will be mapped strongly negative and the **zero inputs** will be mapped near zero in the tanh graph  
        - **differentiable**  
        - **monotonic** --> but derivative is not monotonic  
        
        __**Both tanh and logistic sigmoid activation functions are used in feed-forward nets**__  
        
        3. ReLU (Rectified Linear Unit)  
        ![Fig: ReLU v/s Logistic Sigmoid](https://miro.medium.com/max/700/1*XxxiA0jJvPrHEJHD4z893g.png)  
        - **range**: `[0, infinity)`  
        - used in almost all the convolutional neural networks or deep learning  
        - half rectified (from bottom) --> $f(z)$ is zero when z is less than zero and $f(z)$ is equal to z when z is above or equal to zero  
        - **monotonic** --> the function and its derivative both are monotonic  
        - issue: all the negative values become zero immediately which decreases the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turns affects the resulting graph by not mapping the negative values appropriately  
        
        4. Leaky ReLU  
        ![Fig : ReLU v/s Leaky ReLU](https://miro.medium.com/max/700/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)  
        - solve the dying ReLU problem  
        - increase the range of the ReLU function. Usually, $a=0.01$  
        - when a is not 0.01 then it is called Randomized ReLU  
        - **range**: `(-infinity, infinity)`  
        - **monotonic** --> both Leaky and Randomized ReLU functions are monotonic in nature --> derivatives also monotonic in nature  
        
        
## Why derivative/differentiation is used ?
When updating the curve, to know in which **direction** and **how much to change or update** the curve depending upon the **slope**.That is why we use differentiation in almost every part of Machine Learning and Deep Learning  
![cheet sheet](https://miro.medium.com/max/700/1*p_hyqAtyI8pbt2kEl6siOQ.png)  
![Derivative of Activation Functions](https://miro.medium.com/max/700/1*n1HFBpwv21FCAzGjmWt1sg.png)      