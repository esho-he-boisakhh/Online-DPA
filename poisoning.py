import numpy as np
from projection import project_onto_box
from utils import compute_recursive_gradient

#poisoning optimizer
def optimize_poison_sample(model,
                           X_eval: np.ndarray,
                           y_eval: np.ndarray,
                           jacobian_history,
                           buffer,
                           sample_index: int,
                           learning_rate: float,
                           poison_lr: float,
                           lower_bound: float,
                           upper_bound: float,
                           current_time_index: int,
                           epsilon: float,
                           max_inner_iter: int,
                           beta: float):
    if buffer.is_empty():
        raise ValueError("Rehearsal buffer is empty")

    # Unpack true stream time index alongside sample data
    x_i, y_i, theta_i, _, time_i = buffer.get_sample(sample_index)
    if theta_i is None:
        raise ValueError("Theta snapshot required")
    # Skip if time index is not strictly less than current (guard for Eq. 14)
    if time_i >= current_time_index:
        raise ValueError("Sample time index must be strictly less than current time")

    x_current = x_i.copy()
    prev_loss = None

    for iteration in range(max_inner_iter):
        # Compute recursive gradient ∂J(D_cache ∪ D_reh_buf, θ_t) / ∂x_i
        # using correct implementation from utils.py (Eq. 16)
        # Pass true stream time index time_i, not buffer position
        recursive_grad = compute_recursive_gradient(
            model=model,
            X_eval=X_eval,
            y_eval=y_eval,
            jacobian_history=jacobian_history,
            x_i=x_current,
            y_i=y_i,
            theta_i=theta_i,
            learning_rate=learning_rate,
            i=time_i,              # true stream time index
            t=current_time_index
        )

        # Step in gradient ascent direction, scaled by α·β^iteration (line 14, Algorithm 1)
        effective_lr = poison_lr * (beta ** iteration)
        x_candidate = x_current + effective_lr * recursive_grad

        # Project onto feasible domain Π (Eq. 6 / line 14)
        x_candidate = project_onto_box(x_candidate, lower_bound, upper_bound)

        # Evaluate outer loss J(D_cache_t, θ_p*)
        y_pred = model.predict(X_eval)
        current_loss = float(np.mean((y_pred - y_eval) ** 2))

        # Convergence check
        if prev_loss is not None:
            if abs(current_loss - prev_loss) < epsilon:
                break
        prev_loss = current_loss
        x_current = x_candidate

    return x_current.astype(np.float64)