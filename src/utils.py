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
