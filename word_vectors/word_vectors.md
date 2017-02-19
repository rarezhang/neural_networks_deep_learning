# word vectors  
- [google implementaion](https://code.google.com/archive/p/word2vec/)  
- vector representation of words  
- vectors can represent words in an abstract way  
- able to capture the relationships between words in an expressive way  
- can use the vectors as inputs to a neural networ  
- vector of weight  
    + 1-of-N (or "one-hot") encoding every element in the vector is associated with a word in the vocabulary  
    ![one-hot](https://cloud.githubusercontent.com/assets/5633774/23104667/9f18c276-f68f-11e6-987c-fec57daff2dd.png)  
    + in word2vec, a distributed representation of a word is used:  
    ![distribution-representation](https://cloud.githubusercontent.com/assets/5633774/23104673/b2cab644-f68f-11e6-8d1d-a0be4c16da5e.png)  
        * Each word is representated by a distribution of weights across those elements.   
        * Instead of a one-to-one mapping between an element in the vector and a word, the representation of a word is spread across all of the elements in the vector, and each element in the vector contributes to the definition of many words.          


### [Efficient Estimation of Word Representations in Vector Space – Mikolov et al. 2013](https://arxiv.org/pdf/1301.3781.pdf)  
![example](https://cloud.githubusercontent.com/assets/5633774/23104747/fe4fba78-f690-11e6-9608-184acb3a1e9d.png)
- Continuous Bag-of-Words model (CBOW)  
![cbow](https://cloud.githubusercontent.com/assets/5633774/23104740/e00e6744-f690-11e6-8390-4b0e2ad5d690.png)
    + input: context words  
        * each word is encoded in __one-hot__ form  
        * V-dimensional vectors with just one of the elements set to one (V: vocabulary size)  
    + hideen layer  
    ![cbow-hidden](https://cloud.githubusercontent.com/assets/5633774/23104807/eebed7d2-f691-11e6-896d-1b71f691a33a.png)  
        * activation function: sum the corresponding __hot__ rows in W1 and dividing by C to take the average  (C: # of input word vectors; W1: weight matrix)  
    + output layer  
        * a selection of a row from W1 based on the input  
        * from the hidden layer to the output layer, the second weight matrix W2 can be used to compute a score for each word in the vocabulary, and softmax can be used to obtain the posterior distribution of words  
    + training objective: maximize the conditional probability of observing the __focus word__ given the input context words  

- Continuous Skip-gram model (skip-gram)  
    + input: the focus word as the single input vector
    + hidden layer: 
        * activation function: amounts to copying the corresponding row from the weights matrix W1 (linear)  
    + output layer: 
        * target context words   
        * C multinomial distributions  
    + training objective: mimimize the summed prediction error across all context words in the output layer      
        



### [Distributed Representations of Words and Phrases and their Compositionality – Mikolov et al. 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- optimisations for the skip-gram model  
    + why: update every output word vector for every word in a training instance is very expensive  
    + hierarchical softmax  
        ![hierarchical-softmax](https://cloud.githubusercontent.com/assets/5633774/23104993/61db2c9a-f695-11e6-9e4a-3657f7903b1c.png)
        * uses a binary tree (binary Huffman tree) to represent all words in the vocabulary  
        * words: leaves in the tree  
        * path: a unique path from the root to the leaf --> to estimate the probability ( probability of a random walk starting from the root ending at the leaf in question) of the word represented by the leaf
    + negative sampling  
        * only update a sample of output words per iteration  
        * the target output word should be kept in the sample and gets updated, and add to this a few (non-target) words as __negative samples__  
        * a probabilistic distribution is needed for the sampling process, and it can be arbitrarily chosen. can determine a good distribution empirically  
        * subsampling: counter the imbalance between rare and frequent words in the training set, each word in the training set is discarded with probability P(wi) where  
        ![subsampling](https://cloud.githubusercontent.com/assets/5633774/23105109/b74d201a-f696-11e6-91c3-586da232162d.png)  

### [Linguistic Regularities in Continuous Space Word Representations – Mikolov et al. 2013](http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf)
- vector-oriented reasoning based on word vectors (King - Man + Woman = Queen)  
![king-queen1](https://cloud.githubusercontent.com/assets/5633774/23104712/56c19416-f690-11e6-8a49-fdc6d6a0ebe8.png)  
![king-queen2](https://cloud.githubusercontent.com/assets/5633774/23104720/602c65da-f690-11e6-9346-28f4aa8209f7.png)  

- good at answering analogy questions (a is to b as c is to ?)  
![analogy](https://cloud.githubusercontent.com/assets/5633774/23104697/1e7768e2-f690-11e6-930d-79ee064fcd46.png)  

### Explanation: 
- [word2vec Parameter Learning Explained – Rong 2014](https://arxiv.org/pdf/1411.2738v3.pdf)
- [word2vec Explained: Deriving Mikolov et al’s Negative Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722v1.pdf)