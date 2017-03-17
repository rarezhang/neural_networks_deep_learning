# Neural Networks and Deep Learning

[Reading notes](http://neuralnetworksanddeeplearning.com/)

### neural nets
- neural networks: __infer rules__  
- neural networks:  
    + two types of artificial neuron:  
        * perceptron  
        * sigmoid neuron  
    + standard learning algorithms: stochastic gradient descent  


    
### perceptron  
- take binary inputs and produces a single binary output (weight up evidence)  
![perceptron](https://cloud.githubusercontent.com/assets/5633774/23443637/6525c374-fded-11e6-9d5a-fbcf13695528.png)  
- input: x1,x2,... 0 or 1  
- output: 0 or 1  
- weights: real numbers expressing the importance of the respective inputs to the output 
- bias == threshold (how easy it is to get the percepron to fire)  
![perceptron_decision_rule1](https://cloud.githubusercontent.com/assets/5633774/23443684/bf192290-fded-11e6-96e8-b9447df47c5f.png)
![perceptron_decision_rule2](https://cloud.githubusercontent.com/assets/5633774/23444169/975eff4c-fdf0-11e6-9b79-a0ca3c3762d7.png)
- if multiple layers perceptrons:
    + first layer: make simple decisions by weighing up the input evidence  
    + xx-th layer: making decision by weighing up the results from previous layers --> can make a decision at a more complex and abstract level      
    ![multilayer_perceptron](https://cloud.githubusercontent.com/assets/5633774/23444026/d0d7fa68-fdef-11e6-9876-b13e302ef278.png)  
- perceptron can be used to simulate NAND gates:  
    + NAND gates are universal for computation --> perceptrons are universal for computation   
    + learning algorithms: automatically tune weights and biases  




### sigmoid neuron  
- goal: small change in any weight causes a small change in the output (perceptrons: completely flip)  
- input:  x1,x2,... [0,1]  
- output: sigmod(w*x+b), sigmod function: ![sigmod](https://cloud.githubusercontent.com/assets/5633774/23445040/e257fe04-fdf5-11e6-9a05-91d96e2f39ba.png)  
    + when z=w*x+b is large and positive --> sigmod(z)~1  
    + when z=w*x+b is large and negative --> sigmod(z)~0  
    + shape: smoothed out version of a step function(perceptron: outputs 1 or 0)  
    ![sigmod_shape](https://cloud.githubusercontent.com/assets/5633774/23445127/67cff6cc-fdf6-11e6-895f-3f63c159ff70.png)  
    
    
### the architecture of neural networks 
![architecture](https://cloud.githubusercontent.com/assets/5633774/23445436/8e116544-fdf8-11e6-99d4-4753a7ce3f55.png)  
- feed-forward neural networks: the output from one layer is used as input to the next layer  
    + no loops  
    + with loops: recurrent neural networks  

    


### gradient descent  
- to solve minimization problems  
- goal: find a set of weights and biases which make the cost as small as possible  
    + minimize the quadratic cost  
    + examine the classification accuracy  
- cost function: e.g., quadratic cost function (MSE)  
![cost_function_MSE](https://cloud.githubusercontent.com/assets/5633774/23446057/4c070b36-fdfd-11e6-9fd5-1abb980ad900.png)  
    + why need cost function: making small changes to the weights and biases won't cause any change in the number of training records classified correctly --> difficult to decide how to change the weights  
- theory  
    ![cost_shape](https://cloud.githubusercontent.com/assets/5633774/23446487/2caf56d6-fe01-11e6-94aa-465026ad9f9d.png)  
    + the change of cost:  
    ![change_of_cost](https://cloud.githubusercontent.com/assets/5633774/23446492/35867de8-fe01-11e6-92c7-2595a58cd25d.png)  
    + gradient vector (relates changes in v to changes in C):  
    ![gradient_vector](https://cloud.githubusercontent.com/assets/5633774/23446507/5d1c25f6-fe01-11e6-9fd8-5137e903b67b.png)  
    + re-write the change of cost:  
    ![re_change_of_cost](https://cloud.githubusercontent.com/assets/5633774/23446528/84c98cb0-fe01-11e6-92b1-6748af073c53.png)  
    + choose the direction by learning rate(small&positve):  
    ![direction](https://cloud.githubusercontent.com/assets/5633774/23446565/d8519864-fe01-11e6-9525-ef4f67e18666.png)  
    + re-write the change of cost(guarantee the change is negative):  
    ![re-cost](https://cloud.githubusercontent.com/assets/5633774/23446589/0b8b6e3a-fe02-11e6-834d-508369477abc.png)  
    + update rule: keep decreasing cost C until reach a global minimum  
    ![move](https://cloud.githubusercontent.com/assets/5633774/23446860/f3363d54-fe03-11e6-9b1d-85b6f0d13306.png)
    ![direction](https://cloud.githubusercontent.com/assets/5633774/23446565/d8519864-fe01-11e6-9525-ef4f67e18666.png)
    ![v](https://cloud.githubusercontent.com/assets/5633774/23446887/2d50d36e-fe04-11e6-848e-10674e6b9099.png)  
    + re-write update rule (gradient vector has corresponding components w and b):  
    ![update_rule](https://cloud.githubusercontent.com/assets/5633774/23447093/c4b426ce-fe05-11e6-9fdc-424ba87a2c03.png)  
- stochastic gradient descent  
    + goal: estimate the gradient by computing a small sample of randomly chosen training inputs  
    + by averaging over the small sample (m), quickly get estimate of the true gradient  
    ![stochastic gradient descent ](https://cloud.githubusercontent.com/assets/5633774/23447243/ccd981cc-fe06-11e6-96e4-71432ea91c5a.png)  ![image](https://cloud.githubusercontent.com/assets/5633774/23447251/db0524a4-fe06-11e6-9260-a83f68ad9f00.png)  
    + stochastic gradient descent works by picking out a randomly chosen mini-batch of training inputs:  
    ![stochastic gradient descent update rule](https://cloud.githubusercontent.com/assets/5633774/23447280/1d25093a-fe07-11e6-9918-7bce555ccd53.png)  
    
    
### simple network [code](https://github.com/rarezhang/neural_networks_deep_learning/blob/master/neural_networks_and_deep_learning/Network.py) 
![simple_network](https://cloud.githubusercontent.com/assets/5633774/23445679/69e65dda-fdfa-11e6-9c86-d4e437970f1c.png)    

    
### matrix-based notation
    
### back-propagation algorithm  
- back-propagation gives us detailed insights into how changing the weights and biases changes the overall behavior of the network 

    
    
    



