"""
optimizer.py - Classical Optimization Engine for Q-AMP
Author: Srijan (Q-AMP Team)

Drives the classical optimization loop using COBYLA via scipy.optimize.minimize
to find the ground-state energy minimum of the Ising Hamiltonian.
"""

from typing import Callable, Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize


def run_qaoa_optimization(
    cost_function: Callable[[np.ndarray], float],
    p: int = 2,
    initial_params: Optional[np.ndarray] = None,
    maxiter: int = 200,
    tol: float = 1e-4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the classical COBYLA optimization loop over QAOA variational parameters.

    Parameters
    ----------
    cost_function : Callable[[np.ndarray], float]
        Callback function evaluating expectation energy <H> for given [gammas, betas].
    p : int, default=2
        Number of QAOA circuit layers (p=2 gives 2*p = 4 total parameters).
    initial_params : np.ndarray, optional
        Starting variational angles. If None, initialized randomly.
    maxiter : int, default=200
        Maximum allowable iterations for the classical optimizer.
    tol : float, default=1e-4
        Convergence tolerance for COBYLA termination.
    verbose : bool, default=True
        If True, prints progress updates during convergence.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'optimal_params': Optimized parameter array [gammas, betas]
        - 'optimal_energy': Ground state energy value (eV)
        - 'history': List of energy values recorded across iterations
        - 'nfev': Total number of quantum circuit evaluations
        - 'success': Boolean indicating if optimizer converged successfully
    """
    num_params = 2 * p

    # Randomly sample initial parameters within valid angular domains if not supplied
    if initial_params is None:
        gamma_init = np.random.uniform(0, 2 * np.pi, size=p)
        beta_init = np.random.uniform(0, np.pi, size=p)
        initial_params = np.concatenate([gamma_init, beta_init])

    energy_history = []
    iteration_counter = 0

    def objective_wrapper(params: np.ndarray) -> float:
        """Wrapper to log energy evaluation history at each iteration step."""
        nonlocal iteration_counter
        energy = float(cost_function(params))
        energy_history.append(energy)
        iteration_counter += 1

        if verbose and (iteration_counter % 10 == 0 or iteration_counter == 1):
            print(f"[COBYLA Iter {iteration_counter:3d}] Current Energy: {energy:8.4f} eV")

        return energy

    # Execute classical minimization using scipy COBYLA
    res = minimize(
        fun=objective_wrapper,
        x0=initial_params,
        method='COBYLA',
        options={
            'maxiter': maxiter,
            'rhobeg': 0.5,
            'tol': tol
        }
    )

    optimal_params = res.x
    optimal_energy = float(res.fun)

    if verbose:
        print("\n" + "="*45)
        print("   Q-AMP COBYLA OPTIMIZATION COMPLETE")
        print("="*45)
        print(f" Minimum Energy Reached : {optimal_energy:.4f} eV")
        print(f" Total Evaluations      : {res.nfev}")
        print(f" Optimal Gammas (γ)     : {np.round(optimal_params[:p], 4)}")
        print(f" Optimal Betas  (β)     : {np.round(optimal_params[p:], 4)}")
        print("="*45 + "\n")

    return {
        'optimal_params': optimal_params,
        'optimal_energy': optimal_energy,
        'history': energy_history,
        'nfev': res.nfev,
        'success': res.success
    }


# Standalone Unit Test Block
if __name__ == "__main__":
    print("--- Running optimizer.py Standalone Test ---")

    # Define a target energy minimum at -4.50 eV for validation
    TARGET_ANGLES = np.array([1.25, 0.85, 0.52, 0.31])
    TARGET_ENERGY = -4.50

    def mock_qaoa_circuit_evaluator(params: np.ndarray) -> float:
        """Simulates QAOA statevector energy landscape quadratic in parameter distance."""
        quadratic_penalty = np.sum((params - TARGET_ANGLES) ** 2)
        return TARGET_ENERGY + float(quadratic_penalty)

    # Test run with p=2 layers
    test_results = run_qaoa_optimization(
        cost_function=mock_qaoa_circuit_evaluator,
        p=2,
        initial_params=np.array([1.0, 1.0, 0.5, 0.5]),
        maxiter=100,
        verbose=True
    )

    print("Standalone Test Results Verification:")
    print("Target Energy expected : -4.50 eV")
    print("Achieved Energy result :", round(test_results['optimal_energy'], 2), "eV")
    assert abs(test_results['optimal_energy'] - TARGET_ENERGY) < 1e-2, "Test failed: Optimizer did not converge!"
    print("\n✓ optimizer.py unit test passed successfully!")
