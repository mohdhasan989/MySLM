import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tokenizer import load_text, tokenize, build_vocab, encode, decode
from scripts.dataset import TextDataset
text = load_text()

print("TEXT:")
print(text)

words = tokenize(text)

print("\nWORDS:")
print(words[:10])

word2id, id2word = build_vocab(words)

print("\nVOCAB:")
print(word2id)

ids = encode(words, word2id)

print("\nIDS:")
print(ids[:10])

decoded = decode(ids, id2word)

print("\nDECODED:")
print(decoded[:10])

data = TextDataset()

print(len(data))

for i in range(5):
    x, y = data[i]
    print(x, "->", y)

print("---- EMBEDDING TEST START ----")
import torch
from scripts.model import EmbeddingModel


model = EmbeddingModel()

x, y = data[0]

x = torch.tensor([x])

print("INPUT:", x)

out = model(x)

print("EMBED SHAPE:", out.shape)
from scripts.model import SelfAttention
from config import EMBED_SIZE

print("---- ATTENTION TEST ----")

att = SelfAttention(EMBED_SIZE)

out2 = att(out)

print("ATT SHAPE:", out2.shape)

from scripts.model import MultiHeadAttention
from config import NUM_HEADS

print("---- MULTIHEAD TEST ----")

mha = MultiHeadAttention(EMBED_SIZE, NUM_HEADS)

out3 = mha(out2)

print("MHA SHAPE:", out3.shape)

from scripts.model import TransformerBlock

print("---- TRANSFORMER BLOCK TEST ----")

block = TransformerBlock(EMBED_SIZE, NUM_HEADS)

out4 = block(out3)

print("BLOCK SHAPE:", out4.shape)

from scripts.model import FullModel
from config import VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS

print("---- FULL MODEL TEST ----")

model2 = FullModel(
    VOCAB_SIZE,
    EMBED_SIZE,
    NUM_HEADS,
    NUM_LAYERS
)

x, y = data[0]

x = torch.tensor([x])

out5 = model2(x)

print("MODEL SHAPE:", out5.shape)

import torch.nn as nn
import torch.optim as optim

from scripts.model import FullModel
from config import VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS


print("---- TRAINING START ----")

model = FullModel(
    VOCAB_SIZE,
    EMBED_SIZE,
    NUM_HEADS,
    NUM_LAYERS
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5


for epoch in range(epochs):

    total_loss = 0

    for i in range(len(data)):

        x, y = data[i]

        x = torch.tensor([x])
        y = torch.tensor([y])

        out = model(x)

        # use last token output
        out = out[:, -1, :]

        loss = criterion(out, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch, "Loss:", total_loss)

print("---- PREDICTION ----")

model.eval()

x, y = data[0]

x = torch.tensor([x])

out = model(x)

out = out[:, -1, :]

pred = torch.argmax(out, dim=-1)

print("Pred id:", pred.item())
print("Target id:", y)

text = load_text()
words = tokenize(text)

word2id, id2word = build_vocab(words)

print("Pred word:", id2word[pred.item()])
print("Target word:", id2word[y])
print("---- GENERATION ----")

model.eval()

text = load_text()
words = tokenize(text)

word2id, id2word = build_vocab(words)


def generate(start_words, max_len=10):

    tokens = start_words.split()

    ids = [word2id[w] for w in tokens if w in word2id]

    for _ in range(max_len):

        x = torch.tensor([ids[-6:]])  # use last SEQ_LEN

        out = model(x)

        out = out[:, -1, :]

        pred = torch.argmax(out, dim=-1).item()

        ids.append(pred)

    result = [id2word[i] for i in ids]

    return " ".join(result)


print(generate("my name"))

print("---- GENERATE FROM INPUT ----")

model.eval()

text = load_text()
words = tokenize(text)

word2id, id2word = build_vocab(words)


def generate_from_input(prompt, max_len=15):

    tokens = prompt.split()

    ids = [word2id.get(w, 0) for w in tokens]

    for _ in range(max_len):

        x = torch.tensor([ids[-6:]])   # use last SEQ_LEN

        out = model(x)

        out = out[:, -1, :]

        pred = torch.argmax(out, dim=-1).item()

        ids.append(pred)

    result = [id2word[i] for i in ids]

    return " ".join(result)


while True:

    s = input("Enter text: ")

    if s == "exit":
        break

    print(generate_from_input(s))
    