#!/usr/bin/python3
# coding=utf-8
"""
implementing word2vec in pytorch
skip-gram model
https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
"""
import numpy as np
import torch
import torch.nn.functional as F


# corpus 

def tokenize_corpus(corpus):
    return [x.split() for x in corpus]



corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


tokenized_corpus = tokenize_corpus(corpus)


# unique tokens 
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}



vocabulary_size = len(vocabulary)



# generate pairs 'center word' and 'context word'
# assume context window to be symmetirc and =2 
window_size = 2
idx_pairs = []
# for each sentence


for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array



# input layer 
# center word encoded in one-hot manner. It dimensions are [1, vocabulary_size]
# one hot encoding 
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x





embedding_dims = 5  # makes v vectors 
# weight matrix 
W1 = torch.randn(embedding_dims, vocabulary_size).float()
W2 = torch.randn(vocabulary_size, embedding_dims).float()
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = get_input_layer(data).float()
        y_true = torch.from_numpy(np.array([target])).long()

        # hidden layer 
        z1 = torch.matmul(W1, x)   # matmul: Matrix product of two tensors.

        # output layer 
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()  # todo 

        # backward pass  
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        # optimization SDG  
        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
