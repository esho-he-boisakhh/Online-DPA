import numpy as np
from config import (
    FEATURE_LOWER_BOUND,
    FEATURE_UPPER_BOUND,
    LABEL_LOWER_BOUND,
    LABEL_UPPER_BOUND
)
from load_dataset import load_ccpp_stream

def _normalize_to_range(values, lower, upper):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.full_like(values, lower)
    normalized = (values - min_val) / (max_val - min_val)
    scaled = normalized * (upper - lower) + lower
    return scaled

def generate_stream(stream_length: int,
                    feature_dim: int,
                    filepath: str = 'Folds5x2_pp.xlsx',
                    noise_std: float = 0.05):
    if stream_length <= 0:
        raise ValueError("stream_length must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    # Load real CCPP dataset
    X_stream, y_stream = load_ccpp_stream(filepath)

    # Trim to requested stream length
    X_stream = X_stream[:stream_length]
    y_stream = y_stream[:stream_length]

    # Normalize features column-wise to [0, 1]
    for col in range(X_stream.shape[1]):
        X_stream[:, col] = _normalize_to_range(
            X_stream[:, col],
            FEATURE_LOWER_BOUND,
            FEATURE_UPPER_BOUND
        )

    # Normalize labels to [0, 1]
    y_stream = _normalize_to_range(
        y_stream,
        LABEL_LOWER_BOUND,
        LABEL_UPPER_BOUND
    )

    # Type casting
    X_stream = X_stream.astype(np.float64)
    y_stream = y_stream.astype(np.float64)

    # Checking
    assert X_stream.shape == (stream_length, feature_dim)
    assert y_stream.shape == (stream_length,)

    return X_stream, y_stream