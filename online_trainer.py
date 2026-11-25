import numpy as np
from typing import Dict, List

class OnlineTrainer:
    #init
    def __init__(self,
                 model,
                 learning_rate: float,
                 track_jacobians: bool = False,
                 track_loss: bool = True):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.model = model
        self.learning_rate = learning_rate
        self.track_jacobians = track_jacobians
        self.track_loss = track_loss
        self.loss_history: List[float] = []
        # Jacobian history keyed by TRUE stream time index (self.step_count
        # at the moment of the call), not by list-append position.
        #
        # FIX: previously this was a flat list, only appended to when
        # batch_size == 1. Once the online attack loop started training
        # on combined multi-sample batches (current cache + rehearsal
        # buffer) -- which happens on nearly every step after warm-up --
        # the list stopped growing while callers kept indexing into it
        # using raw time indices, causing either silent misalignment
        # (wrong Jacobian retrieved) or IndexError (silently swallowed
        # by callers' try/except blocks), and starving the recursive
        # gradient computation almost immediately after warm-up.
        #
        # Now a jacobian is recorded on EVERY train_batch call, using
        # the correct Jacobian for whatever update actually happened:
        # the single-sample Jacobian for batch_size==1, or the batch
        # (mean-gradient) Jacobian otherwise. Keys are always
        # contiguous (0, 1, 2, ...) as long as train_batch is called
        # exactly once per real stream time step, which is how
        # online_dpa.py's main loop uses it.
        self.jacobian_history: Dict[int, np.ndarray] = {}
        self.step_count = 0

    #train - 1 batch
    def train_batch(self,
                    X_batch: np.ndarray,
                    y_batch: np.ndarray) -> Dict:
        batch_size = X_batch.shape[0]
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        theta_before = self.model.theta.copy()
        if batch_size == 1:
            gradient = self.model.gradient_single(
                X_batch[0],
                y_batch[0]
            )
        else:
            gradient = self.model.gradient_batch(
                X_batch,
                y_batch
            )
        #param update
        self.model.update_parameters(gradient, self.learning_rate)
        theta_after = self.model.theta.copy()

        #loss
        if batch_size == 1:
            prediction = self.model.predict(X_batch[0])
            loss = (prediction - y_batch[0]) ** 2
        else:
            loss = self.model.compute_mse(X_batch, y_batch)
        if self.track_loss:
            self.loss_history.append(float(loss))

        jacobian = None
        # Track a Jacobian for EVERY step now, not just batch_size==1
        # steps. Use the single-sample formula when batch_size==1
        # (identical to the previous, paper-matching behaviour), and
        # the batch (mean-gradient) formula otherwise, which is the
        # correct parameter-recurrence Jacobian for whatever
        # multi-sample update was actually just applied.
        if self.track_jacobians:
            if batch_size == 1:
                jacobian = self.model.jacobian_theta_update(
                    X_batch[0],
                    self.learning_rate
                )
            else:
                jacobian = self.model.jacobian_batch_theta_update(
                    X_batch,
                    self.learning_rate
                )
            # Keyed by true time index (this call's step_count), so
            # lookups by absolute time index are always correct.
            self.jacobian_history[self.step_count] = jacobian

        self.step_count += 1

        return {
            "loss": float(loss),
            "gradient": gradient.copy(),
            "jacobian": jacobian,
            "theta_before": theta_before,
            "theta_after": theta_after
        }

    #full stream
    def train_stream(self,
                     X_stream: np.ndarray,
                     y_stream: np.ndarray,
                     batch_size: int):
        n_samples = X_stream.shape[0]
        if n_samples != y_stream.shape[0]:
            raise ValueError("X_stream and y_stream length mismatch")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for i in range(0, n_samples, batch_size):
            X_batch = X_stream[i:i + batch_size]
            y_batch = y_stream[i:i + batch_size]
            self.train_batch(X_batch, y_batch)
        return self.loss_history

    #reset
    def reset(self):
        self.loss_history = []
        self.jacobian_history = {}
        self.step_count = 0