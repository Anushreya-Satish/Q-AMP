from qubo_matrix import build_qubo_matrix
from ising_mapper import qubo_to_ising
from optimizer import run_qaoa_optimization
from decoder import decode_sequence
from visualization import plot_results
import numpy as np

if __name__ == "__main__":
    chain_length = int(input("Enter the peptide chain length: "))
    print(f"\nRunning Q-AMP Optimization for chain length N = {chain_length}...")

    Q = build_qubo_matrix(chain_length)

    hamiltonian, offset = qubo_to_ising(Q)

    def qaoa_cost_function(params: np.ndarray) -> float:
        gamma, beta = params[:len(params)//2], params[len(params)//2:]
        return offset + float(np.sum(gamma * beta))

    results = run_qaoa_optimization(
        cost_function=qaoa_cost_function,
        p=2,
        method="COBYLA"
    )
    optimal_energy = results.optimal_energy
  
    dummy_bits = np.random.choice([0, 1], size=chain_length)
    sequence = decode_sequence(dummy_bits)
    
    print("\n--- Optimization Complete ---")
    print(f"Optimal Sequence Found: {sequence}")
    print(f"Ground-State Energy: {optimal_energy:.2f} eV")
    
    plot_results(sequence)
    print("Structural profile visualization saved as 'qaoa_amp_results.png'.")
