from config import DATA_PATH


def load_text():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text


def tokenize(text):
    words = text.split()
    return words


def build_vocab(words, vocab_size=1000):
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)
    vocab = sorted_words[:vocab_size]

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    return stoi, itos


def encode(words, stoi):
    return [stoi[w] for w in words if w in stoi]


def decode(ids, itos):
    return [itos[i] for i in ids]