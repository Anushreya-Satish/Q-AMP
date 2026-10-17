"""
qaoa_circuit.py - QAOA Variational Circuit Builder for Q-AMP
Author: Shibasis Chatterjee (Q-AMP Team)

Builds the QAOA statevector evolution circuit for the antimicrobial peptide
design problem. Applies p alternating layers of cost and mixer unitaries:

    |gamma, beta> = PROD_{k=1..p} [ U(H_M, beta_k) U(H_C, gamma_k) ] |+>^(x)N

The cost Hamiltonian H_C is diagonal (built only from Z and ZZ terms by
ising_mapper.py), so U(H_C, gamma) = exp(-i*gamma*H_C) is implemented
exactly with RZ and RZZ rotations - no Trotter approximation needed.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def extract_ising_terms(hamiltonian):
    """
    Reads the Z and ZZ coefficients out of a SparsePauliOp.

    Returns
    -------
    linear : dict {qubit_index: coefficient}      for Z_i terms
    quadratic : dict {(i, j): coefficient}        for Z_i Z_j terms
    constant : float                              for the identity term
    """
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError("hamiltonian must be a qiskit SparsePauliOp")

    linear, quadratic, constant = {}, {}, 0.0

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        coeff = float(np.real(coeff))
        n = len(label)

        # Qiskit labels are little-endian: label[-1] is qubit 0
        z_positions = [n - 1 - k for k, ch in enumerate(label) if ch == "Z"]
        unsupported = [ch for ch in label if ch not in ("I", "Z")]

        if unsupported:
            raise ValueError(
                f"Term '{label}' contains non-diagonal Pauli operators. "
                "The cost Hamiltonian must contain only I, Z and ZZ terms."
            )

        if len(z_positions) == 0:
            constant += coeff
        elif len(z_positions) == 1:
            i = z_positions[0]
            linear[i] = linear.get(i, 0.0) + coeff
        elif len(z_positions) == 2:
            i, j = sorted(z_positions)
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + coeff
        else:
            raise ValueError(
                f"Term '{label}' couples {len(z_positions)} qubits. "
                "Only 1- and 2-body Z terms are supported."
            )

    return linear, quadratic, constant


def build_qaoa_circuit(hamiltonian, params, p=2, measure=False):
    """
    Programmatically constructs the p-layer QAOA circuit.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        Diagonal Ising cost Hamiltonian from ising_mapper.qubo_to_ising().
    params : array-like, length 2*p
        Variational angles, ordered [gamma_1..gamma_p, beta_1..beta_p]
        (this matches the ordering used by optimizer.py).
    p : int
        Number of QAOA layers (circuit depth).
    measure : bool
        If True, append measurement of all qubits into a classical register.

    Returns
    -------
    QuantumCircuit
    """
    params = np.asarray(params, dtype=float)
    if params.size != 2 * p:
        raise ValueError(f"Expected {2 * p} parameters for p={p}, got {params.size}")

    num_qubits = hamiltonian.num_qubits
    gammas, betas = params[:p], params[p:]
    linear, quadratic, _ = extract_ising_terms(hamiltonian)

    qc = QuantumCircuit(num_qubits)

    # Initial state: uniform superposition |+>^(x)N
    qc.h(range(num_qubits))

    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]

        # --- Cost unitary: exp(-i * gamma * H_C) ---
        # RZ(theta)  = exp(-i*theta*Z/2)    -> theta = 2*gamma*coeff
        # RZZ(theta) = exp(-i*theta*ZZ/2)   -> theta = 2*gamma*coeff
        for i, coeff in linear.items():
            qc.rz(2.0 * gamma * coeff, i)
        for (i, j), coeff in quadratic.items():
            qc.rzz(2.0 * gamma * coeff, i, j)

        qc.barrier()

        # --- Mixer unitary: exp(-i * beta * sum_i X_i) ---
        qc.rx(2.0 * beta, range(num_qubits))

        if layer < p - 1:
            qc.barrier()

    if measure:
        qc.measure_all()

    return qc


def cost_diagonal(hamiltonian, offset=0.0):
    """
    Builds the energy of every basis state as a 1-D array of length 2^N,
    WITHOUT constructing the dense 2^N x 2^N matrix.

    Index convention: entry k holds the energy of basis state |k>, where
    bit i of k is the value of variable x_i (qiskit little-endian).
    """
    linear, quadratic, constant = extract_ising_terms(hamiltonian)
    num_qubits = hamiltonian.num_qubits
    num_states = 2 ** num_qubits

    indices = np.arange(num_states)

    # spin[i] = +1 when x_i = 0, -1 when x_i = 1
    spins = 1.0 - 2.0 * np.stack(
        [(indices >> i) & 1 for i in range(num_qubits)]
    ).astype(float)

    energies = np.full(num_states, constant + offset, dtype=float)
    for i, coeff in linear.items():
        energies += coeff * spins[i]
    for (i, j), coeff in quadratic.items():
        energies += coeff * spins[i] * spins[j]

    return energies


def get_probabilities(hamiltonian, params, p=2):
    """Runs the circuit and returns the measurement probability of each state."""
    qc = build_qaoa_circuit(hamiltonian, params, p=p)
    return np.asarray(Statevector(qc).probabilities())


def qaoa_expectation(params, hamiltonian, p=2, offset=0.0, diagonal=None):
    """
    Evaluates <gamma, beta| H_C |gamma, beta> + offset.

    This is the cost function consumed by optimizer.run_qaoa_optimization().
    Pass a precomputed `diagonal` to avoid rebuilding it on every iteration.
    """
    if diagonal is None:
        diagonal = cost_diagonal(hamiltonian, offset=offset)
    probs = get_probabilities(hamiltonian, params, p=p)
    return float(np.dot(probs, diagonal))


def make_cost_function(hamiltonian, p=2, offset=0.0):
    """
    Returns a single-argument function f(params) -> energy, ready to hand
    straight to optimizer.run_qaoa_optimization(cost_function=...).

    The energy diagonal is computed once here and reused across all
    optimizer iterations.
    """
    diagonal = cost_diagonal(hamiltonian, offset=offset)

    def cost_function(params):
        return qaoa_expectation(params, hamiltonian, p=p, diagonal=diagonal)

    return cost_function


def sample_ground_state(hamiltonian, params, p=2):
    """
    Returns (bitstring_list, probability) for the most probable state.

    bitstring_list[i] is the binary value of variable x_i, ready for
    decoder.decode_sequence().
    """
    probs = get_probabilities(hamiltonian, params, p=p)
    best = int(np.argmax(probs))
    bits = [(best >> i) & 1 for i in range(hamiltonian.num_qubits)]
    return bits, float(probs[best])


if __name__ == "__main__":
    print("--- qaoa_circuit.py self-test ---")

    # Minimal 3-qubit Ising problem (independent of the other modules)
    H = SparsePauliOp.from_list([("IIZ", -0.5), ("IZI", -0.5), ("ZZI", 0.25)])
    p = 2
    angles = np.array([0.6, 0.3, 0.9, 0.4])

    qc = build_qaoa_circuit(H, angles, p=p)
    print(f"Qubits: {qc.num_qubits}   Depth: {qc.depth()}")
    print(f"Gate counts: {dict(qc.count_ops())}")

    probs = get_probabilities(H, angles, p=p)
    print(f"Probabilities sum to {probs.sum():.10f}  (must be 1.0)")
    assert abs(probs.sum() - 1.0) < 1e-10, "State is not normalised!"

    # Zero angles must leave the state in uniform superposition
    uniform = get_probabilities(H, np.zeros(2 * p), p=p)
    expected = 1.0 / 2 ** H.num_qubits
    print(f"Zero-angle probability per state: {uniform[0]:.6f} (expected {expected:.6f})")
    assert np.allclose(uniform, expected), "Zero angles should give |+>^N!"

    # Diagonal must agree with the dense matrix
    dense = np.real(np.diag(H.to_matrix()))
    fast = cost_diagonal(H)
    print(f"Diagonal matches dense matrix: {np.allclose(dense, fast)}")
    assert np.allclose(dense, fast), "cost_diagonal disagrees with H.to_matrix()!"

    # Expectation must sit within the Hamiltonian's spectrum
    e = qaoa_expectation(angles, H, p=p)
    print(f"Expectation energy: {e:.4f} eV   (spectrum {fast.min():.2f} .. {fast.max():.2f})")
    assert fast.min() - 1e-9 <= e <= fast.max() + 1e-9, "Energy outside spectrum!"

    bits, prob = sample_ground_state(H, angles, p=p)
    print(f"Most probable state: x = {bits}  (p = {prob:.4f})")

    print("\nAll qaoa_circuit.py self-tests passed.")
