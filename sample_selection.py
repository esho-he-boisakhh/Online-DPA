import numpy as np
from typing import Tuple
from utils import compute_recursive_gradient

#selecting the sample w most influence
def select_most_influential_sample(model,
                                   X_eval: np.ndarray,
                                   y_eval: np.ndarray,
                                   jacobian_history,
                                   buffer,
                                   learning_rate: float,
                                   current_time_index: int,
                                   current_x: np.ndarray,
                                   current_y: float):
    if buffer.is_empty():
        raise ValueError("Buffer is empty")

    best_score = -np.inf
    best_index = None
    best_source = None

    # 1. Evaluate buffer samples
    for i in range(len(buffer)):
        x_i, y_i, theta_i, _, time_i = buffer.get_sample(i)
        if theta_i is None:
            continue
        # Skip if time index is not strictly less than current (Eq. 14 case i >= t)
        if time_i >= current_time_index:
            continue
        try:
            recursive_grad = compute_recursive_gradient(
                model=model,
                X_eval=X_eval,
                y_eval=y_eval,
                jacobian_history=jacobian_history,
                x_i=x_i,
                y_i=y_i,
                theta_i=theta_i,
                learning_rate=learning_rate,
                i=time_i,          # true stream time index, not buffer position
                t=current_time_index
            )
        except Exception:
            continue

        grad_norm = float(np.linalg.norm(recursive_grad, ord=2))
        buffer.update_influence(i, grad_norm)

        if grad_norm > best_score:
            best_score = grad_norm
            best_index = i
            best_source = "buffer"

    # 2. Evaluate current sample (Ct)
    try:
        # use model.theta.copy() directly — model.get_parameters() does not exist
        theta_current = model.theta.copy()
        recursive_grad_current = compute_recursive_gradient(
            model=model,
            X_eval=X_eval,
            y_eval=y_eval,
            jacobian_history=jacobian_history,
            x_i=current_x,
            y_i=current_y,
            theta_i=theta_current,
            learning_rate=learning_rate,
            i=current_time_index - 1,
            t=current_time_index
        )
        grad_norm_current = float(np.linalg.norm(recursive_grad_current, ord=2))
        if grad_norm_current > best_score:
            best_score = grad_norm_current
            best_index = None
            best_source = "current"
    except Exception:
        pass

    if best_source is None:
        raise RuntimeError("No valid influential sample found")

    return best_index, best_score, best_source