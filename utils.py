import numpy as np
from typing import List

#∂F(θ_t) / ∂θ_t
def compute_outer_objective_gradient(model,
                                     X_eval: np.ndarray,
                                     y_eval: np.ndarray) -> np.ndarray:
    if X_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("Evaluation set size mismatch")
    predictions = model.predict(X_eval)
    errors = predictions - y_eval
    grad = (2.0 / X_eval.shape[0]) * (X_eval.T @ errors)
    return grad.astype(np.float64)

#∂θ_{i+1} / ∂x_i
def compute_theta_wrt_x(x_i: np.ndarray,
                        y_i: float,
                        theta_i: np.ndarray,
                        learning_rate: float) -> np.ndarray:
    k = x_i.shape[0]
    e_i = float(np.dot(theta_i, x_i) - y_i)
    term1 = np.outer(theta_i, x_i)      # θ_i x_i^T
    term2 = e_i * np.eye(k)             # e_i I
    # Correct: -2α(θ_i x_i^T + e_i I) — factor of 2 from MSE gradient (Eq. 15)
    derivative = -2 * learning_rate * (term1 + term2)
    return derivative.astype(np.float64)

#chain jacobian
def compute_jacobian_chain(jacobian_history,
                           start_index: int,
                           end_index: int) -> np.ndarray:
    """
    Compute the product J_{end_index} @ J_{end_index-1} @ ... @ J_{start_index}
    where jacobian_history maps TRUE stream time index -> Jacobian matrix.

    FIX: previously jacobian_history was a flat list and this function
    indexed it by raw list position (jacobian_history[j]). Once the
    caller stopped appending an entry on every real time step (e.g.
    during combined-batch training), list position no longer matched
    true time index j, so this either read the wrong Jacobian silently
    or raised an IndexError that got swallowed by a try/except further
    up the call stack. jacobian_history is now a dict keyed by true
    time index (populated every step by OnlineTrainer), so lookups
    here are always correct, and a missing key now raises a clear,
    informative error instead of failing silently or silently
    misaligning.
    """
    if start_index > end_index:
        raise ValueError("Invalid chain indices")

    if len(jacobian_history) == 0:
        raise ValueError("jacobian_history is empty — no Jacobians recorded yet")

    # Determine feature dimension from any available entry
    any_jacobian = next(iter(jacobian_history.values()))
    dim = any_jacobian.shape[0]

    chain = np.eye(dim)
    for j in range(end_index, start_index - 1, -1):
        if j not in jacobian_history:
            raise KeyError(
                f"Missing Jacobian for time index {j}. Expected a Jacobian "
                f"to be recorded for every real stream step; check that "
                f"OnlineTrainer.train_batch is being called exactly once "
                f"per time step and that track_jacobians=True."
            )
        chain = chain @ jacobian_history[j]
    return chain.astype(np.float64)

#recursive gradient
def compute_recursive_gradient(model,
                               X_eval: np.ndarray,
                               y_eval: np.ndarray,
                               jacobian_history: List[np.ndarray],
                               x_i: np.ndarray,
                               y_i: float,
                               theta_i: np.ndarray,
                               learning_rate: float,
                               i: int,
                               t: int) -> np.ndarray:
    if i >= t:
        raise ValueError("i must be strictly less than t")

    #shape here (k,)
    outer_grad = compute_outer_objective_gradient(
        model,
        X_eval,
        y_eval
    )

    #chain pdt J_{i+2} ... J_t
    if i + 1 <= t - 1:
        chain = compute_jacobian_chain(
            jacobian_history,
            start_index=i + 1,
            end_index=t - 1
        )
    else:
        chain = np.eye(len(theta_i))

    #∂θ_{i+1} / ∂x_i
    theta_wrt_x = compute_theta_wrt_x(
        x_i,
        y_i,
        theta_i,
        learning_rate
    )

    recursive_grad = outer_grad @ chain @ theta_wrt_x
    return recursive_grad.astype(np.float64)

    '''Full multiplication
    outer_grad (1 x k)
    chain (k x k)
    theta_wrt_x (k x k)
    '''