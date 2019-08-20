import torch
import torch.nn as nn


class Code2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super().__init__()
        
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.do = nn.Dropout(dropout)
        
    def forward(self, starts, paths, ends):
        # starts = paths = ends = [batch size, max length], e.g. [[0, 1, 2], [0, 5, 4]], batch size 2, max_length 3

        # W = [batch size, embedding dim, embedding dim * 3]
        W = self.W.repeat(starts.shape[0], 1, 1)
        
        # embedded_* = [batch size, max length, embedding dim]
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        # c = [batch size, max length, embedding dim * 3]
        c = self.do(torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2))
        
        # c = [batch size, embedding dim * 3, max length]
        c = c.permute(0, 2, 1)  # matrix transpose

        # x = [batch size, embedding dim, max length]
        x = torch.tanh(torch.bmm(W, c))
        
        # x = [batch size, max length, embedding dim]
        x = x.permute(0, 2, 1)
        
        # a = [batch size, embedding dim, 1]
        a = self.a.repeat(starts.shape[0], 1, 1)
        
        # z = [batch size, max length]
        z = torch.bmm(x, a).squeeze(2)
        
        # z = [batch size, max length]
        z = torch.softmax(z, dim=1)
        
        # z = [batch size, max length, 1]
        z = z.unsqueeze(2) # attention weights
        
        # x = [batch size, embedding dim, max length]
        x = x.permute(0, 2, 1)
        
        # v = [batch size, embedding dim]
        v = torch.bmm(x, z).squeeze(2)
        
        # out = [batch size, output dim]
        out = self.out(v)
        out = torch.softmax(out, dim=1)

        return out

    def get_code_vec(self, starts, paths, ends):
        # starts = paths = ends = [batch size, max length]
        
        W = self.W.repeat(starts.shape[0], 1, 1)

        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = c.permute(0, 2, 1)
        x = torch.tanh(torch.bmm(W, c))
        x = x.permute(0, 2, 1)

        a = self.a.repeat(starts.shape[0], 1, 1)
        z = torch.bmm(x, a).squeeze(2)
        z = torch.softmax(z, dim=1)
        z = z.unsqueeze(2)
        x = x.permute(0, 2, 1)

        # v = [batch size, embedding dim]
        v = torch.bmm(x, z).squeeze(2)

        return v


class Code2VecIgnoreVal(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super().__init__()

        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, starts, paths, ends):
        # starts = paths = ends = [batch size, max length], e.g. [[0, 1, 2], [0, 5, 4]], batch size 2, max_length 3

        # W = [batch size, embedding dim, embedding dim * 3]
        W = self.W.repeat(starts.shape[0], 1, 1)

        # embedded_* = [batch size, max length, embedding dim]
        embedded_paths = self.path_embedding(paths)

        # c = [batch size, max length, embedding dim]
        c = embedded_paths

        # c = [batch size, embedding dim, max length]
        c = c.permute(0, 2, 1)  # matrix transpose

        # x = [batch size, embedding dim, max length]
        x = torch.tanh(torch.bmm(W, c))

        # x = [batch size, max length, embedding dim]
        x = x.permute(0, 2, 1)

        # a = [batch size, embedding dim, 1]
        a = self.a.repeat(starts.shape[0], 1, 1)

        # z = [batch size, max length]
        z = torch.bmm(x, a).squeeze(2)

        # z = [batch size, max length]
        z = torch.softmax(z, dim=1)

        # z = [batch size, max length, 1]
        z = z.unsqueeze(2)  # attention weights

        # x = [batch size, embedding dim, max length]
        x = x.permute(0, 2, 1)

        # v = [batch size, embedding dim]
        v = torch.bmm(x, z).squeeze(2)

        # out = [batch size, output dim]
        out = self.out(v)
        out = torch.softmax(out, dim=1)

        return out

    def get_code_vec(self, starts, paths, ends):
        # starts = paths = ends = [batch size, max length]

        W = self.W.repeat(starts.shape[0], 1, 1)
        embedded_paths = self.path_embedding(paths)
        c = embedded_paths
        c = c.permute(0, 2, 1)
        x = torch.tanh(torch.bmm(W, c))
        x = x.permute(0, 2, 1)

        a = self.a.repeat(starts.shape[0], 1, 1)
        z = torch.bmm(x, a).squeeze(2)
        z = torch.softmax(z, dim=1)
        z = z.unsqueeze(2)
        x = x.permute(0, 2, 1)

        # v = [batch size, embedding dim]
        v = torch.bmm(x, z).squeeze(2)

        return v
