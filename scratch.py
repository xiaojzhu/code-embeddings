import torch
import torch.nn as nn

node_embedding = nn.Embedding(3, 5)
path_embedding = nn.Embedding(4, 5)  # embedding dimension 5

starts = torch.tensor([[1,2,1], [0,2,0]])
paths = torch.tensor([[1,2,1], [1,1,2]])
ends = torch.tensor([[1,2,0], [1,1,0]]) # batch size 2 with max_length 3

embedded_starts = node_embedding(starts)
embedded_paths = path_embedding(paths)
embedded_ends = node_embedding(ends)

c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
dropout = nn.Dropout(0)
c = dropout(c)

W = nn.Parameter(torch.randn(1, 5, 3*5))
W = W.repeat(starts.shape[0], 1, 1)
a = nn.Parameter(torch.randn(1, 5, 1))
a = a.repeat(starts.shape[0], 1, 1)

c = c.permute(0, 2, 1)  # matrix transpose
x = torch.tanh(torch.bmm(W, c))
x = x.permute(0, 2, 1)
z = torch.bmm(x, a).squeeze(2)

# z = [batch size, max length]
z = torch.softmax(z, dim=1)

# z = [batch size, max length, 1]
z = z.unsqueeze(2)

# x = [batch size, embedding dim, max length]
x = x.permute(0, 2, 1)

# v = [batch size, embedding dim]
v = torch.bmm(x, z).squeeze(2)

# out = [batch size, vocab_size]
linear = nn.Linear(5, 10)
out = linear(v)
out = torch.softmax(out, dim=1)