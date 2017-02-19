# word vectors 
- [google implementaion](https://code.google.com/archive/p/word2vec/)  
- vector representation of words  
- a vector can represent a word in an abstract way  
- vector of weight  
    + 1-of-N (or "one-hot") encoding every element in the vector is associated with a word in the vocabulary  
    + in word2vec, a distributed representation of a word is used:  
        * Each word is representated by a distribution of weights across those elements.   
        * Instead of a one-to-one mapping between an element in the vector and a word, the representation of a word is spread across all of the elements in the vector, and each element in the vector contributes to the definition of many words.  

### [Efficient Estimation of Word Representations in Vector Space – Mikolov et al. 2013](https://arxiv.org/pdf/1301.3781.pdf)
- Continuous Bag-of-Words model (CBOW)  

- Continuous Skip-gram model (skip-gram) 



### [Distributed Representations of Words and Phrases and their Compositionality – Mikolov et al. 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- optimisations for the skip-gram model  
    + hierarchical softmax  
    + negative sampling  

### [Linguistic Regularities in Continuous Space Word Representations – Mikolov et al. 2013](http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf)
- vector-oriented reasoning based on word vectors (King - Man + Woman = Queen)

### Explanation: 
- [word2vec Parameter Learning Explained – Rong 2014](https://arxiv.org/pdf/1411.2738v3.pdf)
- [word2vec Explained: Deriving Mikolov et al’s Negative Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722v1.pdf)