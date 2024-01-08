import numpy as np
from scipy import sparse


def initialize_sparse_orderly_weights(N, K, p, iss, rc_connectivity, seed, bias=True, dtype="float32"):

    # Input matrix:
    q = int(N / (K + int(bias)))  # Adjust for the extra bias neuron
    Win = np.zeros(shape=(N, K + int(bias)), dtype=dtype)  # Increase the size for the bias term
    #print(Win.shape)
    for i in range(K+int(bias)):
        Win[i * q: (i + 1) * q, i] = iss * (np.random.rand(1, q)[0] - 0.5)

    # Reservoir adjacency matrix:
    # Produce sparse matrix. Unfortunately it must be dense for the operations to work. I'm sure there's a way to
    # capitalise on the sparse operations.
    W = sparse.rand(N, N, density=rc_connectivity, random_state=seed, dtype=dtype).todense()
    # Scale spectral radius:
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    # Divide by current spectral radius, multiply by desired one.
    W *= p / sr

    return Win, W


def initialize_uniform_weights(N, K, p, iss, in_connectivity, rc_connectivity, seed, bias, dtype="float32") -> tuple:

    # Input vector
    random_state = np.random.RandomState(seed=seed)
    Win = np.random.rand(N, K + int(bias)) - 0.5
    # Introduce sparsity (Optional):
    mask = np.random.rand(N, Win.shape[1])  # We want this bounded between [0, 1].
    Win[mask > in_connectivity] = 0
    # Apply input scaling:
    Win *= iss

    # Reservoir adjacency matrix:
    # Produce sparse matrix. Unfortunately it must be dense for the operations to work. I'm sure there's a way to
    # capitalise on the sparse operations. You can subtract 0.5 from the line below to center the distribution.
    W = sparse.rand(N, N, density=rc_connectivity, random_state=seed).todense()
    # Scale spectral radius:
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    # Divide by current spectral radius, multiply by desired one.
    W *= p / sr

    return Win, W
