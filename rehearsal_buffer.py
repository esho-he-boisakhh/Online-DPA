import numpy as np
from typing import Optional, Tuple, List

class RehearsalBuffer:
    #init
    def __init__(self,
                 max_size: int,
                 store_theta_snapshot: bool = True):
        self.influence_scores = []
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.store_theta_snapshot = store_theta_snapshot
        self.features: List[np.ndarray] = []
        self.labels: List[float] = []
        self.theta_snapshots: List[np.ndarray] = []
        # Store original stream time index for each sample
        # needed for correct recursive gradient computation
        self.time_indices: List[int] = []

    def __len__(self) -> int:
        return len(self.features)

    def is_full(self) -> bool:
        return len(self.features) >= self.max_size

    def is_empty(self) -> bool:
        return len(self.features) == 0

    def add_sample(self,
                   x: np.ndarray,
                   y: float,
                   theta_snapshot: Optional[np.ndarray] = None,
                   influence_score: float = 0.0,
                   time_index: int = 0):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be numpy array")
        if not np.isscalar(y):
            raise TypeError("y must be scalar")
        if self.store_theta_snapshot and theta_snapshot is None:
            raise ValueError(
                "theta_snapshot required when store_theta_snapshot=True"
            )
        x = x.astype(np.float64).copy()
        y = float(y)

        if self.is_full():
            # Evict sample with smallest influence score (Algorithm 2 lines 5-6)
            idx = int(np.argmin(self.influence_scores))
            self.features[idx] = x
            self.labels[idx] = y
            if self.store_theta_snapshot:
                self.theta_snapshots[idx] = theta_snapshot.astype(np.float64).copy()
            self.influence_scores[idx] = float(influence_score)
            self.time_indices[idx] = int(time_index)
        else:
            self.features.append(x)
            self.labels.append(y)
            if self.store_theta_snapshot:
                self.theta_snapshots.append(
                    theta_snapshot.astype(np.float64).copy()
                )
            self.influence_scores.append(float(influence_score))
            self.time_indices.append(int(time_index))

    #retrieval — now also returns time_index
    def get_sample(self, index: int) -> Tuple[np.ndarray, float, Optional[np.ndarray], float, int]:
        if index < 0 or index >= len(self.features):
            raise IndexError("Buffer index out of range")
        x = self.features[index].copy()
        y = self.labels[index]
        if self.store_theta_snapshot:
            theta_snapshot = self.theta_snapshots[index].copy()
        else:
            theta_snapshot = None
        return x, y, theta_snapshot, self.influence_scores[index], self.time_indices[index]

    def update_influence(self, index: int, new_score: float):
        self.influence_scores[index] = float(new_score)

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.is_empty():
            return None, None, None
        X = np.vstack(self.features)
        y = np.array(self.labels, dtype=np.float64)
        if self.store_theta_snapshot:
            theta_snapshots = np.vstack(self.theta_snapshots)
        else:
            theta_snapshots = None
        return X, y, theta_snapshots

    #clear
    def clear(self):
        self.features = []
        self.labels = []
        self.theta_snapshots = []
        self.influence_scores = []
        self.time_indices = []