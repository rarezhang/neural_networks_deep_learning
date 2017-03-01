# Neural Networks and Deep Learning

[Reading notes](http://neuralnetworksanddeeplearning.com/)

### neural nets
- neural networks: __infer rules__  
- neural networks:  
    + two types of artificial neuron:  
        * perceptron  
        * sigmoid neuron  
    + standard learning algorithms: stochastic gradient descent  


    
### perceptrons  
- take binary inputs and produces a single binary output (weight up evidence)  
![perceptron](https://cloud.githubusercontent.com/assets/5633774/23443637/6525c374-fded-11e6-9d5a-fbcf13695528.png)  
- output: 0 or 1  
- weights: real numbers expressing the importance of the respective inputs to the output 
- bias == threshold (how easy it is to get the percepron to fire)  
![perceptron_decision_rule1](https://cloud.githubusercontent.com/assets/5633774/23443684/bf192290-fded-11e6-96e8-b9447df47c5f.png)
![perceptron_decision_rule2](https://cloud.githubusercontent.com/assets/5633774/23444169/975eff4c-fdf0-11e6-9b79-a0ca3c3762d7.png)
- if multiple layers perceptrons:
    + first layer: make simple decisions by weighing up the input evidence  
    + xxth layer: making decision by weighing up the results from previous layers --> can make a decision at a more complex and abstract level      
    ![multilayer_perceptron](https://cloud.githubusercontent.com/assets/5633774/23444026/d0d7fa68-fdef-11e6-9876-b13e302ef278.png)  
- percepron can be used to simulate NAND gates:  
    + NAND gates are universal for computaion --> perceptrons are universal for computation   
    + learning algorithms: automatically tune weights and biases  




### sigmoid neuron  