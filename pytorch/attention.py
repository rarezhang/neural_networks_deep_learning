#!/usr/bin/python3
# coding=utf-8
"""
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math, copy, time
import matplotlib.pyplot as plt 
import seaborn 
seaborn.set_context(context='talk')


# model architecture 

class EncoderDecoder(nn.Module):
    """
    a standard encoder-decoder architecture 
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        take in and process masked src and target sequences 
        """
        return self.decode(self.encode(src, src_mask), src_mask. tgt, tgt_mask)
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        
        
        
        
class Generator(nn.Module):
    """
    define standard linear + softmax generation step
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
    
# encoder and decoder stacks 
## encoder     
    
def clones(module, N):
    """
    produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class Encoder(nn.Module):
    """
    core encoder is a stack of N layers
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        pass the input (and mask) through each layer in turn 
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


## employ a residula connection around each of the two sub-layers, followed by layer normalization 
class LayerNorm(nn.Module):
    """
    construct a layernorm module 
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2
    
class SublayerConnection(nn.Module):
    """
    a residule connection followed by a layer norm 
    note for code simplicity the norm is first as opposed to last 
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    """
    encoder si made up of two sub-layers: the first is a multi-head self-attention mechanism; the second is a simple, position-wise fully connected feed-forward network
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](s, self.feed_forward)
        
        
## decoder
class Decoder(nn.Module):
    """
    generic N layer decoder with masking
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    """
    decoder si made fo self-attn, src-attn, and feed forward"
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size 
        self.self_attn = self_attn 
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
def subsequent_mask(size):
    """
    mask out subsequent positions:
    modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    return torch.from_numpy(subsequent_mask) == 0
    

# attention 
"""
mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
"""
def attention(query, key, value, mask=None, dropout=None):
    """
    SCALED DOT PRODUCT ATTENTION"
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
        '''
        Fills elements of self tensor with value where mask is one. The shape of mask must be broadcastable with the shape of the underlying tensor.
        mask (ByteTensor) – the binary mask
        value (float) – the value to fill in with
        '''
    p_attn = F.softmax(scores, dim=-1)    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn    
    
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        take in model size and number of heads
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # assue d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if maks is not None:
            # same mask applied to all h heads 
            mask = mask.unsqueeze(1) # specify where we want add or remove an axis.
        nbatches = query.size(0)
        
    # 1. do all the linear projections in batch from d_model ==> h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
    
    # 2. apply attention on all the projected vectors in batch 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
    # 3. `concat` using a view and apply a final linear 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        # contiguous: returns a contiguous tensor containing the same data as self 
        return self.linears[-1](x)  # a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    
    
# position-wise feed-forward network
"""
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between
"""
class PositionwiseFeedForward(nn.Module):
    """
    implements FFN equation
    FFN(x) = max(0, xW1+b1)W2+b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



# embeddings and softmax
"""
- use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model
- use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities
- share the same weight matrix between the two embedding layers and the pre-softmax linear transformation 
- in the embedding layers, multiply those weights by sqrt(d_model)
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
        # lut: lookup table 


# positional encoding 
"""
- in order for the model to make use of `the order of the sequence`
- inject some information about the relative or absolute position of the tokens in the sequence
- add positional encodings to the `input embeddings` at the bottoms of the encoder and decoder stacks
- use sine and cosine functions of different frequencies
- allow the model to easily learn to attend by relative positions
- apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # compute the positonal encodings once in log space
        pe = torch.zeros(max_len, d_model) # pe -> positional encoding
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        '''
        https://github.com/harvardnlp/annotated-transformer/issues/25
        '''
        # div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        '''
        use register_buffer when:
            - you want a stateful part of your model that is not a parameter, but you want it in your state_dict
            - registered buffers are Tensors (not Variables)
        '''
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
        
        
# full model 
"""
define a function that takes in hyperparameters and produces a full model
"""
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    construct a model from hyperparameters
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
            
    # initialize parametrers with Glorot / fan_avg (Glorot uniform initializer, also called Xavier uniform initializer. It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            '''
            Fills the input Tensor with values according to the method described in Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution. 
            Also known as Glorot initialization.
            tensor – an n-dimensional torch.Tensor
            gain – an optional scaling factor
            https://pytorch.org/docs/master/nn.init.html
            '''
    return model        


# training 
## batches and masking 
class Batch:
    """
    Object for holding a batch of data with mask during training:
    a batch object that holds the src and target sentences for training, as well as constructing the masks.
    """
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # unsqueeze(-2) -> increase 1 dimension on the last but two
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
            
        @staticmethod
        def make_std_mask(tgt, pad):
            """
            create a mask to hide padding and future words 
            """
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask
    
## training loop 
"""
- a generic training and scoring function to keep track of loss. 
- pass in a generic loss compute function that also handles parameter updates.
"""
def run_epoch(data_iter, model, loss_compute):
    """
    standard training and logging function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.batch.ntokens
        if i% 50 ==1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss/batch.ntokens, tokens/elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# training data and batching 
"""
- use torch text for batching.
- create batches in a torchtext function that ensures our batch size padded to the maximum batchsize does not surpass a threshold (25000 if we have 8 gpus)
"""
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """
    keep augmenting batch and calculate total number of tokens + padding
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg)+2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
    
    
## Optimizer
class NoamOpt:
    """
    optim wrapper that implements rate
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()    
        
    def rate(self, step=None):
        """
        implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))        