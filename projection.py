import numpy as np

def project_onto_box(x: np.ndarray,
                     lower_bound: float,
                     upper_bound: float) -> np.ndarray:
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be <= upper_bound")

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")

    projected = np.clip(x, lower_bound, upper_bound)

    return projected.astype(np.float64)

def project_batch_onto_box(X: np.ndarray,
                           lower_bound: float,
                           upper_bound: float) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be 2D array")

    projected = np.clip(X, lower_bound, upper_bound)

    return projected.astype(np.float64)

#L2 ball??
# ============================================================
# L2 BALL PROJECTION (OPTIONAL BUT RESEARCH-SAFE)
# ============================================================

def project_onto_l2_ball(x: np.ndarray,
                         center: np.ndarray,
                         radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")

    if x.shape != center.shape:
        raise ValueError("Shape mismatch between x and center")

    diff = x - center
    norm = np.linalg.norm(diff, ord=2)

    if norm <= radius:
        return x.astype(np.float64)

    projected = center + (diff / norm) * radius
    return projected.astype(np.float64)