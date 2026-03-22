import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import VOCAB_SIZE, EMBED_SIZE
import math


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, embed_size)

        for pos in range(max_len):
            for i in range(0, embed_size, 2):

                pe[pos, i] = math.sin(pos / (10000 ** (i / embed_size)))

                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_size)))

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x


class EmbeddingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_SIZE
        )

    def forward(self, x):

        x = self.embedding(x)

        return x
    
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_size):
        super().__init__()

        self.embed_size = embed_size

        # Q K V layers
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(self.embed_size)

        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads

        self.head_dim = embed_size // num_heads

        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)

        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # ---- CAUSAL MASK ----
        mask = torch.tril(torch.ones(T, T)).to(x.device)

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(weights, V)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.fc(out)

        return out
class TransformerBlock(nn.Module):

    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_size, num_heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )

    def forward(self, x):

        # attention
        attn = self.att(x)

        x = self.norm1(x + attn)

        # feed forward
        ff = self.ff(x)

        out = self.norm2(x + ff)

        return out

class FullModel(nn.Module):

    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.pos = PositionalEncoding(embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = self.pos(x)

        for layer in self.layers:
            x = layer(x)

        out = self.fc(x)

        return out