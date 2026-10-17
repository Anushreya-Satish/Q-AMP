
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubo_to_ising(Q):
    Q = np.array(Q, dtype=float)
    n = Q.shape[0]

    if Q.shape[0] != Q.shape[1]:
        raise ValueError("QUBO matrix must be square")

    # Convert binary variables x in {0, 1}
    # into spin variables z in {-1, +1}
    linear = np.zeros(n)
    quadratic = {}

    for i in range(n):
        linear[i] += Q[i, i] / 2

        for j in range(i + 1, n):
            coeff = Q[i, j] + Q[j, i]
            if coeff != 0:
                linear[i] += coeff / 4
                linear[j] += coeff / 4
                quadratic[(i, j)] = coeff / 4

    # Construct the Ising Hamiltonian
    terms = []
    for i in range(n):
        if linear[i] != 0:
            pauli = ["I"] * n
            pauli[n - 1 - i] = "Z"
            terms.append(("".join(pauli), linear[i]))

    for (i, j), coeff in quadratic.items():
        pauli = ["I"] * n
        pauli[n - 1 - i] = "Z"
        pauli[n - 1 - j] = "Z"
        terms.append(("".join(pauli), coeff))

    if not terms:
        return SparsePauliOp("I" * n, coeffs=[0.0]), 0.0

    hamiltonian = SparsePauliOp.from_list(terms)

    # Constant energy offset
    offset = sum(Q[i, i] / 2 for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            offset += (Q[i, j] + Q[j, i]) / 4

    return hamiltonian, offset


if __name__ == "__main__":
    Q = [
        [1, -1],
        [-1, 2]
    ]

    H, offset = qubo_to_ising(Q)

    print("Ising Hamiltonian:")
    print(H)
    print("Constant offset:", offset)



