import numpy as np
from qubo_matrix import build_qubo_matrix         
from ising_mapper import qubo_to_ising, to_sparse_pauli_op  
from qaoa_circuit import make_cost_function, cost_diagonal, sample_ground_state 
from optimizer import run_qaoa_optimization       
from decoder import decode_sequence             
from visualization import plot_results            

def main():
    print("==================================================")
    print("   Q-AMP: Quantum Antimicrobial Peptide Designer  ")
    print("==================================================")
    print(" [INFO] Classical simulation bounds: Recommended N")
    print("        is between 4 and 12 (due to 2^N scaling).")
    print("--------------------------------------------------")

    try:
        chain_length = int(input("Enter peptide chain length N (default 6, max 12): ") or 6)
        if chain_length > 12:
            print(" ⚠️ Warning: N > 12 exceeds efficient classical statevector memory limits (2^N).")
            print("    Automatically adjusting N to 12 for safe local simulation.")
            chain_length = 12
        elif chain_length < 2:
            chain_length = 2
    except ValueError:
        print(" ⚠️ Invalid input detected. Defaulting chain length N = 6.")
        chain_length = 6

    try:
        p_layers = int(input("Enter QAOA circuit depth p (default 2): ") or 2)
        if p_layers < 1:
            p_layers = 1
    except ValueError:
        print(" ⚠️ Invalid input detected. Defaulting circuit depth p = 2.")
        p_layers = 2
        
    print(f"\n[1/5] Building QUBO matrix for N = {chain_length}...")
    Q = build_qubo_matrix(chain_length)
    
    print("[2/5] Mapping QUBO to Ising Hamiltonian...")
    h, J, offset = qubo_to_ising(Q)
    hamiltonian = to_sparse_pauli_op(h, J, offset)
    
    energies = cost_diagonal(hamiltonian)
    print(f"      Spectrum bounds: min = {energies.min():.2f} eV, max = {energies.max():.2f} eV")
    
    print(f"[3/5] Running classical optimizer (depth p={p_layers})...")
    cost_function = make_cost_function(hamiltonian, p=p_layers)
    
    best_result = None
    for seed in range(5):
        res = run_qaoa_optimization(
            cost_function=cost_function, 
            p=p_layers, 
            method="COBYLA", 
            maxiter=300, 
            seed=seed, 
            verbose=False
        )
        if best_result is None or res.optimal_energy < best_result.optimal_energy:
            best_result = res
            
    print(f"      Optimization converged! Ground-State Energy: {best_result.optimal_energy:.4f} eV")
    
    print("[4/5] Sampling optimal bitstring and decoding peptide sequence...")
    bits, probability = sample_ground_state(hamiltonian, best_result.optimal_params, p=p_layers)
    sequence = decode_sequence(bits)
    
    print("\n--------------------------------------------------")
    print("                   RESULTS                        ")
    print("--------------------------------------------------")
    print(f"Optimal Bitstring    : {bits}")
    print(f"Decoded Sequence     : {sequence}")
    print(f"Ground-State Energy  : {best_result.optimal_energy:.4f} eV")
    print(f"State Probability    : {probability:.4f}")
    print("--------------------------------------------------\n")
    
    print("[5/5] Generating structural profile visualization...")
    plot_results(sequence)
    print("Visualization successfully saved as 'qaoa_amp_results.png'.")

if __name__ == "__main__":
    main()
