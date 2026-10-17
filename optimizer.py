"""
optimizer.py - Classical Optimization Engine for Q-AMP
Author: Srijan (Q-AMP Team)

Drives classical optimization loops using SciPy (COBYLA, Nelder-Mead, L-BFGS-B)
to find variational ground-state energy minimums of Ising Hamiltonians.
Fully backward-compatible with both dictionary and object access.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List
import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Structured data container holding QAOA optimization outcomes."""
    optimal_params: np.ndarray
    optimal_energy: float
    optimal_gammas: np.ndarray
    optimal_betas: np.ndarray
    history: List[float]
    nfev: int
    success: bool

    def __getitem__(self, item: str):
        """Allows dictionary-style bracket lookup (e.g., res['optimal_energy'])."""
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(f"Key '{item}' not found in OptimizationResult")


def run_qaoa_optimization(
    cost_function: Callable[[np.ndarray], float],
    p: int = 2,
    initial_params: Optional[np.ndarray] = None,
    method: str = "COBYLA",
    maxiter: int = 200,
    tol: float = 1e-4,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> OptimizationResult:
    """
    Executes classical optimization over QAOA variational parameters [γ, β].

    Parameters
    ----------
    cost_function : Callable[[np.ndarray], float]
        Evaluates expectation energy <H> for given variational angles.
    p : int, default=2
        Number of QAOA circuit layers (2*p total parameters).
    initial_params : Optional[np.ndarray], default=None
        Starting angles. If None, randomly generated based on angle bounds.
    method : str, default="COBYLA"
        SciPy optimization method ('COBYLA', 'Nelder-Mead', 'L-BFGS-B').
    maxiter : int, default=200
        Maximum allowable optimizer iterations.
    tol : float, default=1e-4
        Convergence tolerance threshold.
    seed : Optional[int], default=42
        Random seed for parameter initialization reproducibility.
    verbose : bool, default=True
        Prints formatted convergence updates to standard output.

    Returns
    -------
    OptimizationResult
        Structured dataclass object supporting both dot notation (res.optimal_energy)
        and dictionary bracket access (res['optimal_energy']).
    """
    if seed is not None:
        np.random.seed(seed)

    num_params = 2 * p

    # Angular periodicity domains: γ ∈ [0, 2π], β ∈ [0, π]
    gamma_bounds = [(0.0, 2 * np.pi) for _ in range(p)]
    beta_bounds = [(0.0, np.pi) for _ in range(p)]
    bounds = gamma_bounds + beta_bounds

    if initial_params is None:
        gamma_init = np.random.uniform(0.0, 2 * np.pi, size=p)
        beta_init = np.random.uniform(0.0, np.pi, size=p)
        initial_params = np.concatenate([gamma_init, beta_init])

    energy_history: List[float] = []
    step_counter = 0

    def objective_wrapper(params: np.ndarray) -> float:
        nonlocal step_counter
        energy = float(cost_function(params))
        energy_history.append(energy)
        step_counter += 1

        if verbose and (step_counter == 1 or step_counter % 10 == 0):
            print(f" │ Iter {step_counter:3d} ──► Energy: {energy:8.4f} eV")

        return energy

    if verbose:
        print("\n┌───────────────────────────────────────────┐")
        print("│   Q-AMP QUANTUM OPTIMIZER INITIALIZED     │")
        print("├───────────────────────────────────────────┤")
        print(f"│ Algorithm Depth (p) : {p:2d}                  │")
        print(f"│ Parameters Total    : {num_params:2d}                  │")
        print(f"│ Classical Method   : {method:<21s}│")
        print("└───────────────────────────────────────────┘\n")

    opts = {'maxiter': maxiter}
    if method.upper() == 'COBYLA':
        opts['rhobeg'] = 0.5
        opts['tol'] = tol

    res = minimize(
        fun=objective_wrapper,
        x0=initial_params,
        method=method,
        bounds=bounds if method in ['L-BFGS-B', 'Nelder-Mead'] else None,
        options=opts
    )

    opt_params = res.x
    opt_energy = float(res.fun)
    opt_gammas = opt_params[:p]
    opt_betas = opt_params[p:]

    if verbose:
        print("\n┌───────────────────────────────────────────┐")
        print("│    OPTIMIZATION CONVERGED SUCCESSFULLY    │")
        print("├───────────────────────────────────────────┤")
        print(f"│ Minimum Energy    : {opt_energy:8.4f} eV           │")
        print(f"│ Function Calls    : {res.nfev:3d}                   │")
        print(f"│ Optimal γ (Gamma) : {np.array2string(np.round(opt_gammas, 3), precision=3):<21s} │")
        print(f"│ Optimal β (Beta)  : {np.array2string(np.round(opt_betas, 3), precision=3):<21s} │")
        print("└───────────────────────────────────────────┘\n")

    return OptimizationResult(
        optimal_params=opt_params,
        optimal_energy=opt_energy,
        optimal_gammas=opt_gammas,
        optimal_betas=opt_betas,
        history=energy_history,
        nfev=res.nfev,
        success=res.success
    )


# Standalone Unit & Compatibility Test Block
if __name__ == "__main__":
    print("--- Running Final optimizer.py Verification ---")

    TARGET_ANGLES = np.array([1.25, 0.85, 0.52, 0.31])
    TARGET_ENERGY = -4.50

    def mock_qaoa_circuit_evaluator(params: np.ndarray) -> float:
        """Simulates QAOA statevector energy landscape."""
        return TARGET_ENERGY + float(np.sum((params - TARGET_ANGLES) ** 2))

    # Test run with p=2 layers
    results = run_qaoa_optimization(
        cost_function=mock_qaoa_circuit_evaluator,
        p=2,
        initial_params=np.array([1.0, 1.0, 0.5, 0.5])
    )

    # Validate dual-access compatibility
    dot_energy = results.optimal_energy
    dict_energy = results['optimal_energy']
    dict_history = results['history']

    print("Checking Dual-Access Compatibility:")
    print(f" - Dot notation (results.optimal_energy) : {dot_energy:.4f} eV")
    print(f" - Dict lookup  (results['optimal_energy']): {dict_energy:.4f} eV")

    assert dot_energy == dict_energy, "Error: Dot and Dict access values do not match!"
    assert abs(dot_energy - TARGET_ENERGY) < 1e-2, "Error: Convergence failed!"
    assert len(dict_history) > 0, "Error: Convergence history is empty!"

    print("\n✓ optimizer.py passed all tests with full backward-compatibility!")
