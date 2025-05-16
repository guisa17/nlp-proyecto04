import random
import json
import os

def generate_toy_data(N=10000, max_digits=2):
    """
    Generamos un dataset de N ejemplos de sumas de enteros
    Cada ejemplo una tupla (train, val): (12+7, 19)
    """
    data = []
    for _ in range(N):
        a = random.randint(0, 10**max_digits - 1)
        b = random.randint(0, 10**max_digits - 1)
        inp = f"{a}+{b}"
        tgt = str(a + b)
        data.append((inp, tgt))

    cut = int(0.8 * N)
    return data[:cut], data[cut:]


def save_toy_data(train_data, val_data, path="data/toy_seq2seq.json"):
    """
    Guardamos el dataset en un archivo JSON
    """
    data = {
        "train": [{"input": inp, "target": tgt} for inp, tgt in train_data],
        "val": [{"input": inp, "target": tgt} for inp, tgt in val_data],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_toy_data(path="data/toy_seq2seq.json"):
    """
    Cargamos el dataset desde un archivo JSON
    """
    with open(path, "r") as f:
        data = json.load(f)
    train_data = [(d["input"], d["target"]) for d in data["train"]]
    val_data = [(d["input"], d["target"]) for d in data["val"]]
    return train_data, val_data


class CharTokenizer:
    """
    Tokenizer para el dataset de toy_seq2seq
    """
    def __init__(self):
        chars    = list("0123456789+")
        specials = ["<pad>", "<sos>", "<eos>"]
        self.idx2char = specials + chars
        self.char2idx = {c:i for i,c in enumerate(self.idx2char)}

        self.vocab_size    = len(self.idx2char)
        self.pad_token_id  = self.char2idx["<pad>"]
        self.sos_token_id  = self.char2idx["<sos>"]
        self.eos_token_id  = self.char2idx["<eos>"]


    def encode(self, s: str) -> list[int]:
        return [self.sos_token_id] + [self.char2idx[c] for c in s] + [self.eos_token_id]


    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            c = self.idx2char[i]
            if c in ("<pad>","<sos>","<eos>"):
                continue
            out.append(c)
        return "".join(out)
