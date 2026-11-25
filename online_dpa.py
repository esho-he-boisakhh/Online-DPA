import numpy as np

from rehearsal_buffer import RehearsalBuffer
from sample_selection import select_most_influential_sample
from poisoning import optimize_poison_sample
from online_trainer import OnlineTrainer

from config import (
    FEATURE_LOWER_BOUND,
    FEATURE_UPPER_BOUND,
    MAX_INNER_OPTI_ITER
)


class OnlineDPA:
    def __init__(self,
                 model,
                 model_lr: float,
                 poison_lr: float,
                 poison_decay: float,
                 gamma: float,
                 rehearsal_size: int,
                 epsilon: float,
                 max_iter: int,
                 batch_size: int,
                 convergence_patience: int = 50,
                 convergence_rel_tol: float = None):
        """
        Parameters
        ----------
        convergence_patience : int
            Number of *consecutive* time steps that must satisfy the
            convergence condition (loss close to running max/min) before
            the loop is allowed to stop early. This prevents a single
            floating-point coincidence (e.g. a duplicate row in the
            dataset producing a loss that happens to match a historical
            extreme to within `epsilon`) from silently truncating the
            attack partway through the stream.
        convergence_rel_tol : float or None
            If provided, the convergence check uses a *relative*
            tolerance (rel_tol * scale) in addition to the absolute
            `epsilon`, which makes the check meaningful across loss
            magnitudes rather than only ever firing on near-exact
            bit-for-bit coincidences. If None, only the absolute
            epsilon from config is used (legacy behaviour, but now
            gated by convergence_patience so it is no longer fragile).
        """

        self.model = model
        self.model_lr = model_lr
        self.poison_lr = poison_lr
        self.poison_decay = poison_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.total_poisoned = 0

        self.convergence_patience = max(1, int(convergence_patience))
        self.convergence_rel_tol = convergence_rel_tol
        self._consecutive_converged_steps = 0

        # poison budget
        self.max_poisoned = int(gamma * max_iter * batch_size)
        self.total_poisoned = 0

        # rehearsal buffer
        self.buffer = RehearsalBuffer(
            max_size=rehearsal_size,
            store_theta_snapshot=True
        )

        # trainer
        self.trainer = OnlineTrainer(
            model=self.model,
            learning_rate=self.model_lr,
            track_jacobians=True,
            track_loss=True
        )

        # tracking
        self.mse_history = []

        # Distinct stream-time-indices that have been poisoned at least
        # once. This replaces the old total_poisoned counter, which only
        # incremented on a narrow, unrelated event (an existing buffer
        # sample winning the influence-selection argmax over the newly
        # arriving sample) and drastically undercounted actual poisoning
        # activity -- every sample sitting in the buffer gets
        # gradient-ascent-updated on every iteration the budget gate is
        # open, regardless of which one "won" that round's selection.
        self._poisoned_time_indices = set()

    def _is_within_tolerance(self, value: float, reference: float) -> bool:
        """
        Check |value - reference| < epsilon, optionally widened by a
        relative tolerance scaled to the magnitude of `reference`. Using
        a relative component avoids the check being satisfied only by
        near-exact floating point coincidences when losses are not
        close to zero, while still using strict `epsilon` behaviour if
        convergence_rel_tol is left as None.
        """
        tol = self.epsilon
        if self.convergence_rel_tol is not None:
            tol = max(tol, self.convergence_rel_tol * abs(reference))
        return abs(value - reference) < tol

    # MAIN LOOP (Algorithm 1)
    def run(self, X_stream: np.ndarray, y_stream: np.ndarray):

        n_samples = X_stream.shape[0]

        mse_max = -np.inf
        mse_min = np.inf

        # Guard against trivial convergence before mse_max/mse_min are
        # meaningfully populated (need at least 2 real training steps)
        convergence_check_ready = False

        # Cache the previous time slice (D_cache_{t-1} in paper)
        X_prev_cache = None
        y_prev_cache = None

        self._consecutive_converged_steps = 0

        for t in range(n_samples):

            # Current sample = D_cache_t
            x_t = X_stream[t]
            y_t = y_stream[t]
            X_batch = x_t.reshape(1, -1)
            y_batch = np.array([y_t])

            # Add current sample to rehearsal buffer before selection
            # Pass true stream time index t for correct gradient computation
            self.buffer.add_sample(
                x=x_t,
                y=y_t,
                theta_snapshot=self.model.theta.copy(),
                influence_score=np.inf,  # sentinel: don't evict until score is computed
                time_index=t
            )

            # Need a previous cache and at least 2 buffer samples to attack
            if X_prev_cache is None or len(self.buffer) < 2:
                # Train normally on current sample, slide cache forward
                train_info = self.trainer.train_batch(X_batch, y_batch)
                X_prev_cache = X_batch.copy()
                y_prev_cache = y_batch.copy()
                self.mse_history.append(train_info["loss"])
                mse_max = max(mse_max, train_info["loss"])
                mse_min = min(mse_min, train_info["loss"])
                continue

            # convergence_check_ready after first real attack-capable step
            convergence_check_ready = True

            # 1: Select most influential sample (Algorithm 2)
            # Budget gate now uses the count of DISTINCT poisoned
            # samples (see _poisoned_time_indices), not the old
            # selection-win counter, so gamma actually throttles
            # poisoning as intended.
            budget_ok = (len(self._poisoned_time_indices) / ((t + 1) * self.batch_size)) < self.gamma

            if budget_ok:
                try:
                    selected_index, _, source = select_most_influential_sample(
                        model=self.model,
                        X_eval=X_batch,
                        y_eval=y_batch,
                        jacobian_history=self.trainer.jacobian_history,
                        buffer=self.buffer,
                        learning_rate=self.model_lr,
                        current_time_index=t,
                        current_x=x_t,
                        current_y=y_t
                    )
                except Exception:
                    selected_index = None
                    source = None

                # 2: Poison ALL samples in rehearsal buffer (lines 13-14, Algorithm 1)
                # Paper: for range x_i in D_reh_buf: x_i = Π(x_i + α·β·∇_{x_i} J)
                for buf_idx in range(len(self.buffer)):
                    try:
                        # Unpack true time index stored in buffer
                        _, _, _, _, time_i = self.buffer.get_sample(buf_idx)
                        poisoned_x = optimize_poison_sample(
                            model=self.model,
                            X_eval=X_batch,
                            y_eval=y_batch,
                            jacobian_history=self.trainer.jacobian_history,
                            buffer=self.buffer,
                            sample_index=buf_idx,
                            learning_rate=self.model_lr,
                            poison_lr=self.poison_lr,
                            lower_bound=FEATURE_LOWER_BOUND,
                            upper_bound=FEATURE_UPPER_BOUND,
                            current_time_index=t,
                            epsilon=self.epsilon,
                            max_inner_iter=MAX_INNER_OPTI_ITER,
                            beta=self.poison_decay
                        )
                        # Write poisoned sample back into buffer
                        _, y_old, theta_old, _, time_old = self.buffer.get_sample(buf_idx)
                        self.buffer.features[buf_idx] = poisoned_x
                        self.buffer.labels[buf_idx] = y_old
                        self.buffer.theta_snapshots[buf_idx] = theta_old
                        self.buffer.time_indices[buf_idx] = time_old

                        # Record this stream sample as having been poisoned
                        # at least once (identity tracked by its true
                        # stream time index, not its current buffer slot).
                        self._poisoned_time_indices.add(time_i)

                    except Exception:
                        pass

                # NOTE: this still records which sample "won" the
                # influence-selection argmax this round. It is kept only
                # as a diagnostic (see `selection_wins` in the returned
                # dict below) -- it is NOT the poisoned-sample count.
                if source == "buffer" and selected_index is not None:
                    self.total_poisoned += 1

            # 3: Inner optimization (line 15, Algorithm 1)
            # Train on D_cache_{t-1} ∪ D_reh_buf (poisoned combined set)
            X_reh, y_reh, _ = self.buffer.get_all()
            X_combined = np.vstack([X_prev_cache, X_reh])
            y_combined = np.concatenate([y_prev_cache, y_reh])
            train_info = self.trainer.train_batch(X_combined, y_combined)

            # Compute losses for convergence check (lines 16-24, Algorithm 1)
            # Loss on current arriving cache D_cache_t (outer objective, Eq. 17)
            loss_on_current = self.model.compute_mse(X_batch, y_batch)
            # Loss on previous cache D_cache_{t-1} (inner objective, Eq. 18)
            loss_on_prev = self.model.compute_mse(X_prev_cache, y_prev_cache)

            self.mse_history.append(loss_on_current)

            # ---------------------------------------------------------
            # FIXED convergence check (lines 16-18, Algorithm 1)
            #
            # The original check compared loss_on_current/loss_on_prev
            # against the *running* mse_max/mse_min using only an
            # absolute epsilon (1e-8). Because mse_max/mse_min are
            # updated AFTER this check on every step, a single step
            # whose loss happens to coincide almost exactly with a
            # historical extreme (e.g. from a duplicate row in the
            # dataset, or benign floating point noise) would trigger
            # immediate termination -- regardless of how much of the
            # stream remained. This made total_poisoned, theta, and
            # runtime effectively non-reproducible between runs of the
            # identical configuration.
            #
            # Fix: require the condition to hold for
            # `convergence_patience` CONSECUTIVE steps before breaking,
            # and optionally widen the tolerance with a relative
            # component so the check reflects genuine stabilisation
            # rather than a one-off numeric coincidence.
            # ---------------------------------------------------------
            if convergence_check_ready and mse_max != -np.inf and mse_min != np.inf:
                outer_converged = self._is_within_tolerance(loss_on_current, mse_max)
                inner_converged = self._is_within_tolerance(loss_on_prev, mse_min)

                if outer_converged and inner_converged:
                    self._consecutive_converged_steps += 1
                else:
                    self._consecutive_converged_steps = 0

                if self._consecutive_converged_steps >= self.convergence_patience:
                    break

            # Update running max/min (lines 19-24, Algorithm 1)
            if loss_on_current > mse_max:
                mse_max = loss_on_current
            if loss_on_prev < mse_min:
                mse_min = loss_on_prev

            # Slide the cache window forward
            X_prev_cache = X_batch.copy()
            y_prev_cache = y_batch.copy()

            # Hard budget stop
            if self.total_poisoned >= self.max_poisoned:
                break

        # FINAL METRICS
        #
        # CANONICAL METRIC: final_mse = MEAN of mse_history over the
        # whole run. A single last-sample squared error is extremely
        # noisy (it depends entirely on which one row happens to be
        # last) and can produce nonsensical comparisons -- e.g. a
        # "poisoned" model scoring BETTER than the clean baseline, or
        # a defense showing >100% MSE reduction, purely from last-point
        # variance. The mean is the stable, representative figure and
        # is also what your original Table 4.1/5.1 numbers were built
        # from (0.019437 etc.) -- so this restores consistency with
        # already-reported results.
        #
        # last_sample_mse is kept only as a secondary diagnostic.
        final_mse = float(np.mean(self.mse_history)) if self.mse_history else 0.0
        last_sample_mse = float(self.mse_history[-1]) if self.mse_history else 0.0

        # total_poisoned is now the count of DISTINCT stream samples
        # poisoned at least once (see _poisoned_time_indices above).
        # selection_wins is the old, narrower counter, kept only as a
        # diagnostic of the influence-selection mechanism itself.
        return {
            "final_mse": final_mse,
            "last_sample_mse": last_sample_mse,
            "mse_history": self.mse_history,
            "total_poisoned": len(self._poisoned_time_indices),
            "selection_wins": self.total_poisoned,
            "mse_max": float(mse_max),
            "mse_min": float(mse_min),
            "steps_run": len(self.mse_history)
        }
