import os
import pandas as pd
import matplotlib.pyplot as plt

from NVAR2 import *

def generate_string_dataset(timesteps=10, dimensions=2):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    dataset = np.empty((timesteps, dimensions), dtype=object)

    for t in range(timesteps):
        for d in range(dimensions):
            dataset[t, d] = f"{alphabet[d]}_{t}"

    return dataset


def generate_integer_dataset(timesteps=10, dimensions=2):
    dataset = np.empty((timesteps, dimensions), dtype=int)

    for t in range(timesteps):
        for d in range(dimensions):
            dataset[t, d] = t + d  # You can modify this expression based on the desired integer sequence

    return dataset


if __name__ == "__main__":

    data = generate_integer_dataset(timesteps=50, dimensions=3)
    print(data)

    nvar = NVAR2(delay=2, strides=1, order=2)
    features = nvar.fit_polynomial(data)
    print(features[2])
