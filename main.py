from qubo_matrix import build_qubo_matrix
from ising_mapper import qubo_to_ising
from optimizer import run_qaoa_optimization
from decoder import decode_sequence
from visualization import plot_results

if __name__ == "__main__":
  chain_length = int(input("Enter the peptide chain length: "))
  print(f"Running Q-AMP Optimization for chain length N = {chain_length}...")
  optimal_bits, optimal_energy = run_qaoa_optimization(chain_length)
  sequence = decode_sequence(optimal_bits)
  print("\n--- Optimization Complete ---")
  print(f"Optimal Sequence Found: {sequence}")
  print(f"Ground-State Energy: {optimal_energy:.2f} eV")
  plot_results(sequence)
  print("Structural profile visualization saved as 'qaoa_amp_results.png'.")
