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
![matrix-based notation: weights](https://cloud.githubusercontent.com/assets/5633774/24052436/b899a2de-0af2-11e7-981e-fa71b8794e55.png)
![matrix-based notation: activation](https://cloud.githubusercontent.com/assets/5633774/24052472/dadf6ce8-0af2-11e7-9c1f-fd80479f1d23.png)  
![activation1](https://cloud.githubusercontent.com/assets/5633774/24052517/f8ac57e0-0af2-11e7-876b-c97f58b398bb.png)  
- to rewrite this in a matrix form:  
    + define a weight matrix for each layer  
    + define a bias vector for each layer  
![activation 2](https://cloud.githubusercontent.com/assets/5633774/24052557/249f682e-0af3-11e7-8b43-c3a18ae0bb52.png)  


 
### back-propagation algorithm  
- back-propagation gives us detailed insights into how changing the weights and biases changes the overall behavior of the network  
- goal of back-propagation: compute the partial derivatives ![partial derivatives](https://cloud.githubusercontent.com/assets/5633774/24053059/e0fb0928-0af4-11e7-98f8-05bb386e9da0.png) of the cost function with respect to any weight or bias in the network  
    + partial derivatives:  
    ![partial derivatives](https://cloud.githubusercontent.com/assets/5633774/24053059/e0fb0928-0af4-11e7-98f8-05bb386e9da0.png)  
    + compute the partial derivatives for a single training example  
    + recover the partial derivatives by averaging over training examples  
    + cost function can be written as as function of the outputs from the neural network (a function of the output activations)  
- intermediate error: the error of neuron **_j_** in layer **_l_**:  
    ![intermediate error](https://cloud.githubusercontent.com/assets/5633774/24054569/f297c57c-0af9-11e7-840e-bf1e1d684636.png)  
    ![compute z](https://cloud.githubusercontent.com/assets/5633774/24054602/108b6782-0afa-11e7-8321-cfd594fe9690.png)
    + back-propagation give a way of computing the intermediate error for every layer; then relating those errors to the quantities of real interest: partial derivatives  
- four fundamental equations behind back-propagation:  
    + by computing **[BP1]** and **[BP2]** : can compute the intermediate error for any layer in the network  
    + compute partial derivatives based on the intermediate error: **[BP3]** and **[BP4]**  
    
    1. **[BP1]** error in the output layer:  
        ![bp1](https://cloud.githubusercontent.com/assets/5633774/24055583/bf25f7b4-0afd-11e7-9720-4e578c00257d.png)  
        + first term: how fast the cost is changing as a function of the j-th output activation  
        ![cost](https://cloud.githubusercontent.com/assets/5633774/24055634/fdfb16fe-0afd-11e7-8aae-146f3803e880.png)  
        ![cost derivatives](https://cloud.githubusercontent.com/assets/5633774/24055642/05c9cac4-0afe-11e7-9d34-43bdb69c6080.png)  
        + second term: how fast the activation function is changing   
        ![compute z](https://cloud.githubusercontent.com/assets/5633774/24054602/108b6782-0afa-11e7-8321-cfd594fe9690.png)  
        ![sigma](https://cloud.githubusercontent.com/assets/5633774/23445040/e257fe04-fdf5-11e6-9a05-91d96e2f39ba.png)   
    2. **[BP2]** error in the next layer  
        ![bp2](https://cloud.githubusercontent.com/assets/5633774/24055745/87f5b580-0afe-11e7-88dc-6cf6857d238d.png)  
        + moving the error backward through the network  
    3. **[BP3]** the rate of change of the cost with respect to any bias in the network  
        ![bp3](https://cloud.githubusercontent.com/assets/5633774/24056538/5f3c9cc8-0b01-11e7-902f-31ceffa8bd4d.png)  
        Proof:  
        + (1)  
        ![proof bp3(1)](https://cloud.githubusercontent.com/assets/5633774/24056761/232462e2-0b02-11e7-9974-c73802f16013.png)  
        + (2)  
        ![proof bp3(2)](https://cloud.githubusercontent.com/assets/5633774/24056783/3b1f73be-0b02-11e7-92db-2e4a51431874.png)  
        + according to (1):  
        ![proof bp3(3)](https://cloud.githubusercontent.com/assets/5633774/24056892/949e3d1c-0b02-11e7-9190-a8241fbb0adf.png)  
        + according to (2):  
        ![proof bp3(4)](https://cloud.githubusercontent.com/assets/5633774/24056982/ec19ee7e-0b02-11e7-97ba-05401546424f.png) --> the definition of intermediate error  
    4. **[BP4]** the rate of change of the cost with respect to any weight in the network         
        ![bp4](https://cloud.githubusercontent.com/assets/5633774/24056550/671e0fb2-0b01-11e7-90f2-766f1d557c79.png)  
        Proof:
        + (3)  
        ![proof bp4(1)](https://cloud.githubusercontent.com/assets/5633774/24057193/c378f3d8-0b03-11e7-821d-a67eca1de945.png)  
        + according to (1):  
        ![proof bp4(2)](https://cloud.githubusercontent.com/assets/5633774/24057278/12b27c80-0b04-11e7-9fd4-d647a824a607.png)  
        + according to (3):  
        ![proof bp4(3)](https://cloud.githubusercontent.com/assets/5633774/24057401/7be425fa-0b04-11e7-9dc9-8cf42ea7c2a1.png)   
- code for back-propagation  
    ```
        def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    ```    

    
    
    



