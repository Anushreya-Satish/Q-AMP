
import numpy as np


def build_qubo_matrix(chain_length):
    """
    Build the QUBO matrix for Q-AMP.

    0 = Lysine (hydrophilic)
    1 = Leucine (hydrophobic)

    Hydrophobic reward = -1.5
    Adjacent hydrophobic penalty = +2.0
    """

    if not isinstance(chain_length, (int, np.integer)):
        raise TypeError("Chain length must be an integer.")

    if chain_length < 1:
        raise ValueError("Chain length must be at least 1.")

    Q = np.zeros((chain_length, chain_length), dtype=float)

    for i in range(chain_length):
        Q[i, i] = -1.5

    for i in range(chain_length - 1):
        Q[i, i + 1] = 2.0

    return Q
