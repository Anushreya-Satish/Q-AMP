"""
decoder.py - turns QAOA output into a readable amino acid sequence.

Encoding: bit 0 -> Leucine (Leu, hydrophobic)
          bit 1 -> Lysine  (Lys, hydrophilic / positively charged)

Bit order matches qaoa_circuit.py (qiskit little-endian):
bits[i] is variable x_i, i.e. residue i of the peptide (residue 0 is first).
"""

import numpy as np

AMINO_MAP = {0: "Lys", 1: "Leu"}


def decode_sequence(bits) -> str:
    """
    Convert bits into a sequence string like 'Leu-Lys-Leu-Lys-Leu-Lys'.

    Accepts a list/array of 0s and 1s ([0, 1, 0, ...]) or a string ('010').
    bits[0] is residue 0. This is what main.py and
    qaoa_circuit.sample_ground_state() hand over.
    """
    if isinstance(bits, str):
        bits = [int(b) for b in bits]
    bits = [int(b) for b in np.asarray(bits).flatten()]

    if any(b not in AMINO_MAP for b in bits):
        raise ValueError(f"Bits must be 0 or 1, got: {bits}")

    return "-".join(AMINO_MAP[b] for b in bits)


def index_to_bits(index: int, n_qubits: int) -> list:
    """State index -> bits, little-endian: bits[i] = value of variable x_i."""
    return [(int(index) >> i) & 1 for i in range(n_qubits)]


def get_best_bits(probabilities, n_qubits: int):
    """
    Pick the most likely state from get_probabilities() output.
    Returns (bits, probability). Same result as qaoa_circuit.sample_ground_state.
    """
    probabilities = np.asarray(probabilities)
    if len(probabilities) != 2 ** n_qubits:
        raise ValueError("probabilities length must be 2**n_qubits")
    best = int(np.argmax(probabilities))
    return index_to_bits(best, n_qubits), float(probabilities[best])


def top_k_sequences(probabilities, n_qubits: int, k: int = 5):
    """Return the k most likely (sequence, probability) pairs, useful for plots."""
    probabilities = np.asarray(probabilities)
    top = np.argsort(probabilities)[::-1][:k]
    return [
        (decode_sequence(index_to_bits(i, n_qubits)), float(probabilities[i]))
        for i in top
    ]


if __name__ == "__main__":
    print(decode_sequence([0, 1, 0, 1, 0, 1]))       # Leu-Lys-Leu-Lys-Leu-Lys
    print(decode_sequence("010101"))                  # same
    print(decode_sequence(np.array([0, 1, 1, 0])))    # numpy input, like main.py

    # State index 42 = binary 101010 -> x_0=0, x_1=1, x_2=0 ... (little-endian)
    probs = np.zeros(64)
    probs[42] = 0.8
    probs[21] = 0.2
    bits, p = get_best_bits(probs, 6)
    print(bits, p, "->", decode_sequence(bits))       # Leu-Lys-Leu-Lys-Leu-Lys
    print(top_k_sequences(probs, 6, k=2))
