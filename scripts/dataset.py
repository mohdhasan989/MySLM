from config import SEQ_LEN, VOCAB_SIZE
from utils.tokenizer import load_text, tokenize, build_vocab, encode


class TextDataset:

    def __init__(self):

        text = load_text()

        words = tokenize(text)

        self.word2id, self.id2word = build_vocab(words, VOCAB_SIZE)

        ids = encode(words, self.word2id)

        self.inputs = []
        self.targets = []

        for i in range(len(ids) - SEQ_LEN):

            x = ids[i:i+SEQ_LEN]
            y = ids[i+SEQ_LEN]

            self.inputs.append(x)
            self.targets.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]