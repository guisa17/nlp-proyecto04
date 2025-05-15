import random


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
