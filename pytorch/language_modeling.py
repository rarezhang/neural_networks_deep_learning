#!/usr/bin/python3
# coding=utf-8
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 


torch.manual_seed(1)
# ------------------test -----------------
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)


# ---------------------
# N-Gram Language modeling 

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()  # word tokens 

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
print(vocab)
print(len(vocab))

word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)



class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def get_word_embedding(self, word):
        print(word)
        word = torch.LongTensor([word_to_ix[word]])
        print(word)
        return self.embeddings(word).view(1,-1)




losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:
# 1. inputs 
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

# 2. zero out the gradients 
        model.zero_grad()
        
# 3. forward pass -> get log probabilities over next words 
        log_probs = model(context_idxs)


# 4. compute loss
        # print(torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

# 5. backward pass & update the gradient 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)
print(losses)

print(model.parameters())
print(model.embeddings.weight[word_to_ix['when']])


w = 'when'
emd = model.get_word_embedding(w)
print(emd)
