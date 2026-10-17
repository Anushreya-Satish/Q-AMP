

from itertools import product

import numpy as np


def qubo_to_ising(Q):
    """
    Convert a QUBO matrix to Ising parameters.

    Parameters
    ----------
    Q : array-like, shape (n, n)
        QUBO matrix (does not need to be symmetric or upper-triangular).

    Returns
    -------
    h : np.ndarray, shape (n,)
        Linear coefficients of the Z_i terms.
    J : dict[(int, int), float]
        Quadratic coefficients of the Z_i Z_j terms, keys with i < j.
    offset : float
        Constant energy shift.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n = Q.shape[0]
    h = np.zeros(n)
    J = {}
    offset = 0.0

    # Diagonal (linear) terms: Q_ii * x_i
    for i in range(n):
        h[i] -= Q[i, i] / 2.0
        offset += Q[i, i] / 2.0

    # Off-diagonal (quadratic) terms: (Q_ij + Q_ji) * x_i * x_j
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j] + Q[j, i]
            if q == 0:
                continue
            J[(i, j)] = q / 4.0
            h[i] -= q / 4.0
            h[j] -= q / 4.0
            offset += q / 4.0

    return h, J, offset


def ising_energy(h, J, offset, z):
    """Energy of a spin configuration z (entries +1 or -1)."""
    z = np.asarray(z, dtype=float)
    energy = offset + float(np.dot(h, z))
    for (i, j), coeff in J.items():
        energy += coeff * z[i] * z[j]
    return energy


def binary_to_spin(x):
    """x in {0,1} -> z in {+1,-1}  (x = (1 - z) / 2, so x=0 -> z=+1)."""
    return 1 - 2 * np.asarray(x)


def spin_to_binary(z):
    """z in {+1,-1} -> x in {0,1}."""
    return ((1 - np.asarray(z)) // 2).astype(int)


def ising_to_pauli_terms(h, J, offset, tol=1e-12):
    """
    Express the Ising Hamiltonian as a list of (pauli_label, coefficient).

    Labels follow Qiskit's little-endian convention (rightmost character is
    qubit 0). The constant offset is returned under the identity label.
    """
    n = len(h)
    terms = [("I" * n, offset)]

    for i, coeff in enumerate(h):
        if abs(coeff) > tol:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            terms.append(("".join(label), float(coeff)))

    for (i, j), coeff in J.items():
        if abs(coeff) > tol:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"
            terms.append(("".join(label), float(coeff)))

    return terms


def to_sparse_pauli_op(h, J, offset):
    """Optional: build a Qiskit SparsePauliOp (requires qiskit)."""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:
        raise ImportError("qiskit is required for to_sparse_pauli_op()") from exc
    return SparsePauliOp.from_list(ising_to_pauli_terms(h, J, offset))


def verify(Q, atol=1e-9):
    """Brute-force check that QUBO and Ising energies agree on every state."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    h, J, offset = qubo_to_ising(Q)

    for bits in product([0, 1], repeat=n):
        x = np.array(bits)
        e_qubo = float(x @ Q @ x)
        e_ising = ising_energy(h, J, offset, binary_to_spin(x))
        if not np.isclose(e_qubo, e_ising, atol=atol):
            return False
    return True


if __name__ == "__main__":
    # Example: small 3-variable QUBO
    Q = np.array([
        [-1.0,  2.0,  0.0],
        [ 0.0, -1.0,  2.0],
        [ 0.0,  0.0, -1.0],
    ])

    h, J, offset = qubo_to_ising(Q)
    print("h      :", h)
    print("J      :", J)
    print("offset :", offset)

    print("\nPauli terms:")
    for label, coeff in ising_to_pauli_terms(h, J, offset):
        print(f"  {coeff:+.4f} * {label}")

    print("\nEnergies match for all states:", verify(Q))
