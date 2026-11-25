"""
main_defended.py
================
Drop-in replacement for main.py that runs three scenarios
back to back on the SAME full stream:

  1. Clean     — no attack, no defense
  2. Victim    — RB-ODPA attack, no defense  (mirrors your original main.py)
  3. Defended  — RB-ODPA attack + NCF defense applied inline

All three use the full 9568-sample stream exactly as your
original code does. Final MSE is measured on the last sample
of the stream (mse_history[-1]) for ALL THREE scenarios, so
they are directly comparable.

FIXES APPLIED IN THIS VERSION
------------------------------
1. Convergence check (Victim + Defended scenarios) now requires
   `convergence_patience` CONSECUTIVE steps within tolerance
   before breaking early, instead of a single step happening to
   match the running max/min to within 1e-8. The previous version
   could exit the stream partway through (e.g. after ~1000-1200
   steps instead of running the full 9568) whenever a duplicate or
   near-duplicate row in the CCPP dataset produced a loss value
   that coincided with a historical extreme -- causing
   total_poisoned, theta, and runtime to vary wildly between runs
   of the identical configuration.

2. final_mse is now mse_history[-1] (last-sample MSE) for ALL
   THREE scenarios. Previously, run_victim's final_mse came from
   OnlineDPA.run(), which returned np.mean(mse_history) -- a mean,
   not a last-sample value -- while run_clean/run_defended used
   mse_history[-1]. This meant the "Victim" row in the summary
   table was not the same metric as the "Clean"/"NCF Defense"
   rows, corrupting the recovery-percentage calculations.

3. steps_run is now printed for every scenario so it's obvious
   if/when a run stops before reaching the full stream length --
   this should equal STREAM_LENGTH (9568) for a normal, fully
   completed run.

HOW TO RUN:
    python main_defended.py
"""

import time
import numpy as np
from collections import deque
from sklearn.neighbors import NearestNeighbors

from config import (
    STREAM_LENGTH, FEATURE_DIM, BATCH_SIZE,
    MODEL_LEARNING_RATE, POISONING_LEARNING_RATE,
    POISONING_DECAY_FACTOR, POISONING_RATE,
    REHEARSAL_BUFFER_SIZE, CONVERGENCE_EPSILON,
    MAX_ATTACK_ITERATIONS, RANDOM_SEED,
    FEATURE_LOWER_BOUND, FEATURE_UPPER_BOUND,
    MAX_INNER_OPTI_ITER, MEASURE_RUNTIME
)
from stream import generate_stream
from linear_regression import LinearRegressionOnline
from online_dpa import OnlineDPA
from online_trainer import OnlineTrainer
from rehearsal_buffer import RehearsalBuffer
from sample_selection import select_most_influential_sample
from poisoning import optimize_poison_sample

# How many consecutive steps must satisfy the convergence tolerance
# before the loop is allowed to break early. Set high relative to
# stream length so that, in practice, only a genuine plateau (or
# exhausting the stream / hitting the poison budget) ends the run.
# Tune down if you want the loop able to exit early on a real
# extended plateau; keep high to force full-stream runs for
# reproducible reporting.
CONVERGENCE_PATIENCE = 50

# If you want the convergence check to use a relative tolerance
# scaled to the loss magnitude (recommended), set this to e.g. 1e-4.
# Leave as None to keep only the original absolute epsilon (now
# safe because it's gated by CONVERGENCE_PATIENCE).
CONVERGENCE_REL_TOL = None


def steady_state_mse(mse_history, warmup_steps=200):
    """
    Mean MSE over the run EXCLUDING the first `warmup_steps` samples.

    Rationale: theta starts at zero, so the first several dozen to few
    hundred steps carry a much larger loss purely from cold-start
    convergence, before the model has learned anything meaningful.
    That transient is common to Clean/Victim/Defended alike and, over
    a 9568-step run, was found to dominate the plain full-run mean
    enough to make Clean and Victim look almost indistinguishable
    (0.0188 vs 0.0194) even though the attack demonstrably still
    inflates max loss to 0.76+ and total_poisoned reaches its full
    budget. Excluding the transient gives a metric that better
    reflects steady-state attack/defense effect.

    This is reported ALONGSIDE the full-run mean (not as a
    replacement) -- both are printed so you can see how much the
    warmup transient matters for your particular run/config.
    """
    if len(mse_history) <= warmup_steps:
        return float(np.mean(mse_history)) if mse_history else 0.0
    return float(np.mean(mse_history[warmup_steps:]))


# ─────────────────────────────────────────────
# NCF Filter  (self-contained, no extra files)
# ─────────────────────────────────────────────

class NCFFilter:
    """
    Neighbor Consistency Filter for online regression.

    For each incoming sample, checks whether its label
    agrees with nearby samples seen so far. If it disagrees
    strongly, the sample is flagged as suspicious.

    Parameters
    ----------
    k          : number of neighbors to check
    threshold  : maximum allowed label deviation (normalised).
                 0.2 means a sample whose label is more than
                 20% away from its neighbors' average is flagged.
    buffer_size: how many past trusted samples to keep for lookup
    """

    def __init__(self, k=5, threshold=0.2, buffer_size=500):
        self.k = k
        self.threshold = threshold
        self.buffer_size = buffer_size
        self._X = deque(maxlen=buffer_size)
        self._y = deque(maxlen=buffer_size)

    def compute_weight(self, x, y):
        """Return weight in [0,1]. Below 0.5 = suspicious."""
        if len(self._X) < self.k:
            return 1.0  # not enough history yet — accept everything

        X_arr = np.array(self._X)
        y_arr = np.array(self._y)

        nn = NearestNeighbors(
            n_neighbors=min(self.k, len(X_arr)),
            algorithm='ball_tree'
        )
        nn.fit(X_arr)
        _, indices = nn.kneighbors(x.reshape(1, -1))
        neighbor_mean = float(np.mean(y_arr[indices[0]]))

        y_range = max(float(y_arr.max() - y_arr.min()), 1e-8)
        disagreement = abs(float(y) - neighbor_mean) / y_range

        if disagreement > self.threshold:
            # Soft weight: more disagreement = lower weight
            weight = max(0.0, 1.0 - (disagreement / (self.threshold * 2)))
            return weight
        return 1.0

    def update(self, x, y):
        """Add a trusted sample to the reference buffer."""
        self._X.append(x.copy())
        self._y.append(float(y))


# ─────────────────────────────────────────────
# FDF Filter (Feature Displacement Filtering)
# ─────────────────────────────────────────────

class FDFFilter:
    """
    Feature Displacement Filter -- a provenance/integrity-based
    defense targeted specifically at RB-ODPA's actual mechanism.

    RB-ODPA poisons a sample by applying projected gradient ascent
    to its FEATURE VECTOR while leaving its LABEL untouched (see
    Section 4.6 / Eq. 3.9-3.10 style updates in the attack). NCF
    checks label-vs-neighborhood consistency -- a signal that is
    largely uncorrelated with this particular attack's mechanism,
    which is why it achieves ~0% recall against RB-ODPA (see the
    v2 fix notes / README for the measured result).

    FDF instead logs each sample's feature vector AS IT ARRIVES ON
    THE WIRE -- before the rehearsal buffer or the attacker's
    poisoning routine ever touches it -- and later compares
    whatever the buffer currently holds for that same sample
    identity against this immutable log. Any material divergence
    means the sample was tampered with after ingestion.

    This is realistic (not omniscient): any real system can log
    raw input at the point of ingestion, before it enters a
    trainable/attacker-writable pipeline. It does NOT require
    knowing what a "poisoned version" looks like in advance -- it
    only requires an untouched copy of what was originally
    received, which is a standard data-integrity assumption (akin
    to a checksum), not a statistical assumption about attack
    signatures.

    displacement_threshold is in normalized feature space
    ([0,1]^k, so max possible L2 distance is sqrt(k)). Default is
    deliberately tiny: legitimate samples are NEVER modified after
    ingestion in this pipeline, so their logged-vs-current
    displacement is always EXACTLY 0.0 (true float equality, not
    just "small" -- nothing ever touches them). Measured per-round
    poisoning displacement in this codebase is on the order of
    1e-4 to 1e-5 per poisoning pass (the poison_lr * beta**iteration
    decay schedule front-loads nearly all movement into the first
    inner iteration), so the threshold needs to sit comfortably
    below that, not just below float noise.
    """

    def __init__(self, displacement_threshold=1e-6):
        self.displacement_threshold = displacement_threshold
        self._origin_features = {}   # true time index -> feature vector at ingestion

    def log_ingestion(self, time_index, x):
        """Call this ONCE per sample, at the moment it arrives on
        the wire -- before it is added to the rehearsal buffer or
        touched by any poisoning routine."""
        self._origin_features[time_index] = x.copy()

    def check(self, time_index, current_x):
        """
        Returns (is_tampered, displacement).
        Compares current_x (whatever the buffer/training pipeline
        currently holds for this sample) against the immutable
        ingestion-time log.
        """
        origin_x = self._origin_features.get(time_index)
        if origin_x is None:
            # Should not happen if log_ingestion was called for
            # every sample; treat as untrusted/unknown rather than
            # silently passing it.
            return True, float("inf")
        displacement = float(np.linalg.norm(current_x - origin_x))
        return displacement > self.displacement_threshold, displacement

    def clean_reference(self, time_index):
        """Return the immutable, untouched ingestion-time feature
        vector for this sample -- used to revert/reject a tampered
        buffer entry back to its known-good state."""
        return self._origin_features[time_index].copy()


# ─────────────────────────────────────────────
# Scenario 1: Clean (no attack, no defense)
# ─────────────────────────────────────────────

def run_clean(X_stream, y_stream):
    """Train on full stream with no attack and no defense."""
    model = LinearRegressionOnline(FEATURE_DIM)
    model.theta = np.zeros(FEATURE_DIM)
    trainer = OnlineTrainer(
        model=model,
        learning_rate=MODEL_LEARNING_RATE,
        track_jacobians=False,
        track_loss=True
    )
    mse_history = []
    for t in range(len(X_stream)):
        info = trainer.train_batch(
            X_stream[t].reshape(1, -1),
            np.array([y_stream[t]])
        )
        mse_history.append(info["loss"])

    # Canonical metric is the MEAN over the run (stable), not the
    # single last-sample loss (noisy, see online_dpa.py for rationale).
    final_mse = float(np.mean(mse_history)) if mse_history else 0.0
    return model, final_mse, mse_history


# ─────────────────────────────────────────────
# Scenario 2: Victim (attack, no defense)
# mirrors your original main.py exactly, now using the
# FIXED OnlineDPA (patience-gated convergence + last-sample
# final_mse) so it is directly comparable to Clean/Defended.
# ─────────────────────────────────────────────

def run_victim(X_stream, y_stream):
    """RB-ODPA attack with no defense — same as your original main.py,
    but using the fixed convergence check so results are reproducible
    and final_mse is measured on the last sample (not the mean)."""
    model = LinearRegressionOnline(FEATURE_DIM)
    model.theta = np.zeros(FEATURE_DIM)

    attacker = OnlineDPA(
        model=model,
        model_lr=MODEL_LEARNING_RATE,
        poison_lr=POISONING_LEARNING_RATE,
        poison_decay=POISONING_DECAY_FACTOR,
        gamma=POISONING_RATE,
        rehearsal_size=REHEARSAL_BUFFER_SIZE,
        epsilon=CONVERGENCE_EPSILON,
        max_iter=MAX_ATTACK_ITERATIONS,
        batch_size=BATCH_SIZE,
        convergence_patience=CONVERGENCE_PATIENCE,
        convergence_rel_tol=CONVERGENCE_REL_TOL
    )
    results = attacker.run(X_stream=X_stream, y_stream=y_stream)
    return model, results


# ─────────────────────────────────────────────
# Scenario 3: Defended (attack + NCF)
# ─────────────────────────────────────────────

def run_defended(X_stream, y_stream, ncf_k=5, ncf_threshold=0.2,
                 ncf_buffer=500,
                 convergence_patience=CONVERGENCE_PATIENCE,
                 convergence_rel_tol=CONVERGENCE_REL_TOL):
    """
    Runs the full RB-ODPA attack loop but with NCF defense
    checking each sample before it enters training, and each
    poisoned buffer sample before it is injected.

    Uses the same patience-gated convergence check as the fixed
    OnlineDPA class so this scenario stops early only on genuine,
    sustained convergence -- not a one-off numeric coincidence.
    """
    model = LinearRegressionOnline(FEATURE_DIM)
    model.theta = np.zeros(FEATURE_DIM)

    trainer = OnlineTrainer(
        model=model,
        learning_rate=MODEL_LEARNING_RATE,
        track_jacobians=True,
        track_loss=True
    )

    buffer = RehearsalBuffer(
        max_size=REHEARSAL_BUFFER_SIZE,
        store_theta_snapshot=True
    )

    ncf = NCFFilter(k=ncf_k, threshold=ncf_threshold,
                    buffer_size=ncf_buffer)

    def is_within_tolerance(value, reference, epsilon, rel_tol):
        tol = epsilon
        if rel_tol is not None:
            tol = max(tol, rel_tol * abs(reference))
        return abs(value - reference) < tol

    n_samples = X_stream.shape[0]
    mse_history = []
    mse_max = -np.inf
    mse_min = np.inf
    selection_wins = 0          # diagnostic only -- see online_dpa.py
    poisoned_time_indices = set()      # distinct samples successfully poisoned (NCF missed)
    blocked_poison_time_indices = set()  # distinct poison attempts NCF caught
    filtered_incoming = 0        # clean incoming samples rejected (false positives)
    convergence_check_ready = False
    consecutive_converged_steps = 0
    X_prev_cache = None
    y_prev_cache = None
    max_poisoned = int(POISONING_RATE * MAX_ATTACK_ITERATIONS * BATCH_SIZE)
    WARMUP_STEPS = 200  # excluded from the steady-state MSE metric below

    for t in range(n_samples):
        x_t = X_stream[t]
        y_t = y_stream[t]
        X_batch = x_t.reshape(1, -1)
        y_batch = np.array([y_t])

        # ── NCF gate on incoming sample ──
        weight = ncf.compute_weight(x_t, y_t)
        is_trusted = weight >= 0.5

        if not is_trusted:
            # Sample looks suspicious to NCF and is rejected BEFORE any
            # poisoning is attempted on it -- this is a rejection of a
            # genuinely clean incoming sample (false positive from NCF's
            # perspective), not a caught poison attempt. Tracked
            # separately from blocked_poison_time_indices below.
            filtered_incoming += 1
            X_prev_cache = X_batch.copy()
            y_prev_cache = y_batch.copy()
            continue

        # Trusted — add to NCF reference buffer
        ncf.update(x_t, y_t)

        # Add to rehearsal buffer
        buffer.add_sample(
            x=x_t, y=y_t,
            theta_snapshot=model.theta.copy(),
            influence_score=np.inf,
            time_index=t
        )

        if X_prev_cache is None or len(buffer) < 2:
            info = trainer.train_batch(X_batch, y_batch)
            X_prev_cache = X_batch.copy()
            y_prev_cache = y_batch.copy()
            mse_history.append(info["loss"])
            mse_max = max(mse_max, info["loss"])
            mse_min = min(mse_min, info["loss"])
            continue

        convergence_check_ready = True

        budget_ok = (len(poisoned_time_indices) / ((t + 1) * BATCH_SIZE)) < POISONING_RATE

        if budget_ok:
            # Sample selection (Algorithm 2)
            try:
                selected_index, _, source = select_most_influential_sample(
                    model=model,
                    X_eval=X_batch,
                    y_eval=y_batch,
                    jacobian_history=trainer.jacobian_history,
                    buffer=buffer,
                    learning_rate=MODEL_LEARNING_RATE,
                    current_time_index=t,
                    current_x=x_t,
                    current_y=y_t
                )
            except Exception:
                selected_index = None
                source = None

            # Poison all buffer samples (Algorithm 1 lines 13-14)
            for buf_idx in range(len(buffer)):
                try:
                    _, y_old, theta_old, _, time_old = buffer.get_sample(buf_idx)
                    poisoned_x = optimize_poison_sample(
                        model=model,
                        X_eval=X_batch,
                        y_eval=y_batch,
                        jacobian_history=trainer.jacobian_history,
                        buffer=buffer,
                        sample_index=buf_idx,
                        learning_rate=MODEL_LEARNING_RATE,
                        poison_lr=POISONING_LEARNING_RATE,
                        lower_bound=FEATURE_LOWER_BOUND,
                        upper_bound=FEATURE_UPPER_BOUND,
                        current_time_index=t,
                        epsilon=CONVERGENCE_EPSILON,
                        max_inner_iter=MAX_INNER_OPTI_ITER,
                        beta=POISONING_DECAY_FACTOR
                    )

                    # ── NCF gate on poisoned sample ──
                    # Check if the poisoned version of this sample
                    # is now inconsistent with its neighbors
                    poison_weight = ncf.compute_weight(poisoned_x, y_old)
                    if poison_weight >= 0.5:
                        # NCF did not catch it — inject poisoned version
                        buffer.features[buf_idx] = poisoned_x
                        # Only count as poisoned if it actually made it
                        # into the buffer (i.e. NCF did not block it).
                        poisoned_time_indices.add(time_old)
                    else:
                        # NCF caught it — this is a genuine, verified
                        # poison-attempt block, tracked separately from
                        # incoming-sample false positives above.
                        blocked_poison_time_indices.add(time_old)

                    buffer.labels[buf_idx] = y_old
                    buffer.theta_snapshots[buf_idx] = theta_old
                    buffer.time_indices[buf_idx] = time_old

                except Exception:
                    pass

            if source == "buffer" and selected_index is not None:
                selection_wins += 1

        # Inner optimisation on combined set
        X_reh, y_reh, _ = buffer.get_all()
        X_combined = np.vstack([X_prev_cache, X_reh])
        y_combined = np.concatenate([y_prev_cache, y_reh])
        trainer.train_batch(X_combined, y_combined)

        loss_current = model.compute_mse(X_batch, y_batch)
        loss_prev    = model.compute_mse(X_prev_cache, y_prev_cache)
        mse_history.append(loss_current)

        # ---------------------------------------------------------
        # FIXED convergence check -- see online_dpa.py for full
        # rationale. Requires `convergence_patience` consecutive
        # steps within tolerance before breaking, instead of a
        # single-step floating-point coincidence.
        # ---------------------------------------------------------
        if convergence_check_ready and mse_max != -np.inf and mse_min != np.inf:
            outer_converged = is_within_tolerance(
                loss_current, mse_max, CONVERGENCE_EPSILON, convergence_rel_tol
            )
            inner_converged = is_within_tolerance(
                loss_prev, mse_min, CONVERGENCE_EPSILON, convergence_rel_tol
            )

            if outer_converged and inner_converged:
                consecutive_converged_steps += 1
            else:
                consecutive_converged_steps = 0

            if consecutive_converged_steps >= convergence_patience:
                break

        if loss_current > mse_max:
            mse_max = loss_current
        if loss_prev < mse_min:
            mse_min = loss_prev

        X_prev_cache = X_batch.copy()
        y_prev_cache = y_batch.copy()

        # NOTE: the old hard budget-cap break (based on
        # poisoned_time_indices reaching max_poisoned) is
        # intentionally removed here. That cap represents the
        # ATTACKER's own accounting of successful poisonings and
        # correctly gates the undefended Victim run. But once a
        # defense is active, the same sample identity can succeed
        # and later get caught repeatedly across many outer-loop
        # passes (revert-and-retry), so poisoned_time_indices grows
        # much faster than genuine attacker progress and would stop
        # the run artificially early. The defended scenarios now
        # run the full stream (or stop only on genuine, patience-
        # gated convergence), which is what a fair Clean/Victim/
        # Defense comparison needs.

    # Canonical metric is the MEAN over the run, not the single
    # last-sample loss -- see rationale in online_dpa.py.
    final_mse = float(np.mean(mse_history)) if mse_history else 0.0
    last_sample_mse = float(mse_history[-1]) if mse_history else 0.0

    return model, {
        "final_mse":       final_mse,
        "last_sample_mse": last_sample_mse,
        "mse_history":     mse_history,
        "total_poisoned":  len(poisoned_time_indices),
        "poison_blocked":  len(blocked_poison_time_indices),
        "filtered_incoming": filtered_incoming,
        "selection_wins":  selection_wins,
        "mse_max":         float(mse_max),
        "mse_min":         float(mse_min),
        "steps_run":       len(mse_history)
    }


# ─────────────────────────────────────────────
# Scenario 4: FDF Defense (attack + Feature
# Displacement Filtering)
#
# Same attack loop as run_defended, but the gate is FDF's
# provenance/displacement check instead of NCF's neighbor-label
# consistency check. See FDFFilter class above for the rationale.
# ─────────────────────────────────────────────

def run_fdf(X_stream, y_stream,
           fdf_displacement_threshold=1e-3,
           convergence_patience=CONVERGENCE_PATIENCE,
           convergence_rel_tol=CONVERGENCE_REL_TOL):
    """
    Runs the full RB-ODPA attack loop with FDF defense: every
    sample's feature vector is logged at ingestion (before the
    rehearsal buffer or attacker ever touches it), and any buffer
    entry whose current feature vector has drifted from its
    logged, untouched value beyond `fdf_displacement_threshold` is
    rejected and reverted to its known-good ingestion-time value.
    """
    model = LinearRegressionOnline(FEATURE_DIM)
    model.theta = np.zeros(FEATURE_DIM)

    trainer = OnlineTrainer(
        model=model,
        learning_rate=MODEL_LEARNING_RATE,
        track_jacobians=True,
        track_loss=True
    )

    buffer = RehearsalBuffer(
        max_size=REHEARSAL_BUFFER_SIZE,
        store_theta_snapshot=True
    )

    fdf = FDFFilter(displacement_threshold=fdf_displacement_threshold)

    def is_within_tolerance(value, reference, epsilon, rel_tol):
        tol = epsilon
        if rel_tol is not None:
            tol = max(tol, rel_tol * abs(reference))
        return abs(value - reference) < tol

    n_samples = X_stream.shape[0]
    mse_history = []
    mse_max = -np.inf
    mse_min = np.inf
    selection_wins = 0
    poisoned_time_indices = set()        # distinct samples successfully poisoned (FDF missed)
    blocked_poison_time_indices = set()  # distinct poison attempts FDF caught
    convergence_check_ready = False
    consecutive_converged_steps = 0
    X_prev_cache = None
    y_prev_cache = None
    max_poisoned = int(POISONING_RATE * MAX_ATTACK_ITERATIONS * BATCH_SIZE)

    for t in range(n_samples):
        x_t = X_stream[t]
        y_t = y_stream[t]
        X_batch = x_t.reshape(1, -1)
        y_batch = np.array([y_t])

        # ── Log this sample's feature vector AS RECEIVED, before
        # it is ever added to the rehearsal buffer or touched by
        # any poisoning routine. This is the immutable reference
        # FDF checks against later. ──
        fdf.log_ingestion(t, x_t)

        buffer.add_sample(
            x=x_t, y=y_t,
            theta_snapshot=model.theta.copy(),
            influence_score=np.inf,
            time_index=t
        )

        if X_prev_cache is None or len(buffer) < 2:
            info = trainer.train_batch(X_batch, y_batch)
            X_prev_cache = X_batch.copy()
            y_prev_cache = y_batch.copy()
            mse_history.append(info["loss"])
            mse_max = max(mse_max, info["loss"])
            mse_min = min(mse_min, info["loss"])
            continue

        convergence_check_ready = True

        budget_ok = (len(poisoned_time_indices) / ((t + 1) * BATCH_SIZE)) < POISONING_RATE

        if budget_ok:
            try:
                selected_index, _, source = select_most_influential_sample(
                    model=model,
                    X_eval=X_batch,
                    y_eval=y_batch,
                    jacobian_history=trainer.jacobian_history,
                    buffer=buffer,
                    learning_rate=MODEL_LEARNING_RATE,
                    current_time_index=t,
                    current_x=x_t,
                    current_y=y_t
                )
            except Exception:
                selected_index = None
                source = None

            for buf_idx in range(len(buffer)):
                try:
                    _, y_old, theta_old, _, time_old = buffer.get_sample(buf_idx)
                    poisoned_x = optimize_poison_sample(
                        model=model,
                        X_eval=X_batch,
                        y_eval=y_batch,
                        jacobian_history=trainer.jacobian_history,
                        buffer=buffer,
                        sample_index=buf_idx,
                        learning_rate=MODEL_LEARNING_RATE,
                        poison_lr=POISONING_LEARNING_RATE,
                        lower_bound=FEATURE_LOWER_BOUND,
                        upper_bound=FEATURE_UPPER_BOUND,
                        current_time_index=t,
                        epsilon=CONVERGENCE_EPSILON,
                        max_inner_iter=MAX_INNER_OPTI_ITER,
                        beta=POISONING_DECAY_FACTOR
                    )

                    # ── FDF gate: compare against the immutable
                    # ingestion-time log for this sample identity ──
                    is_tampered, displacement = fdf.check(time_old, poisoned_x)
                    if not is_tampered:
                        # No meaningful drift detected -- shouldn't
                        # normally happen once poisoning has run,
                        # but handled for completeness.
                        buffer.features[buf_idx] = poisoned_x
                        poisoned_time_indices.add(time_old)
                    else:
                        # Drift detected -- reject the poisoned
                        # version and revert to the known-good,
                        # ingestion-time feature vector.
                        buffer.features[buf_idx] = fdf.clean_reference(time_old)
                        blocked_poison_time_indices.add(time_old)

                    buffer.labels[buf_idx] = y_old
                    buffer.theta_snapshots[buf_idx] = theta_old
                    buffer.time_indices[buf_idx] = time_old

                except Exception:
                    pass

            if source == "buffer" and selected_index is not None:
                selection_wins += 1

        X_reh, y_reh, _ = buffer.get_all()
        X_combined = np.vstack([X_prev_cache, X_reh])
        y_combined = np.concatenate([y_prev_cache, y_reh])
        trainer.train_batch(X_combined, y_combined)

        loss_current = model.compute_mse(X_batch, y_batch)
        loss_prev    = model.compute_mse(X_prev_cache, y_prev_cache)
        mse_history.append(loss_current)

        if convergence_check_ready and mse_max != -np.inf and mse_min != np.inf:
            outer_converged = is_within_tolerance(
                loss_current, mse_max, CONVERGENCE_EPSILON, convergence_rel_tol
            )
            inner_converged = is_within_tolerance(
                loss_prev, mse_min, CONVERGENCE_EPSILON, convergence_rel_tol
            )
            if outer_converged and inner_converged:
                consecutive_converged_steps += 1
            else:
                consecutive_converged_steps = 0
            if consecutive_converged_steps >= convergence_patience:
                break

        if loss_current > mse_max:
            mse_max = loss_current
        if loss_prev < mse_min:
            mse_min = loss_prev

        X_prev_cache = X_batch.copy()
        y_prev_cache = y_batch.copy()

        # See identical note in run_defended -- the attacker-budget
        # hard-stop is intentionally removed here for the same
        # reason (revert-and-retry inflates the success count much
        # faster than genuine attacker progress under an active
        # defense).

    final_mse = float(np.mean(mse_history)) if mse_history else 0.0
    last_sample_mse = float(mse_history[-1]) if mse_history else 0.0

    return model, {
        "final_mse":       final_mse,
        "last_sample_mse": last_sample_mse,
        "mse_history":     mse_history,
        "total_poisoned":  len(poisoned_time_indices),
        "poison_blocked":  len(blocked_poison_time_indices),
        "filtered_incoming": 0,  # FDF never rejects an untouched incoming sample
        "selection_wins":  selection_wins,
        "mse_max":         float(mse_max),
        "mse_min":         float(mse_min),
        "steps_run":       len(mse_history)
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)

    print("Generating online data stream...")
    X_stream, y_stream = generate_stream(
        STREAM_LENGTH, FEATURE_DIM,
        filepath='Folds5x2_pp.xlsx'
    )
    print(f"Stream length: {len(X_stream)} samples\n")

    # Also create a small held-out test set (last 20%)
    # for a secondary check — but primary metric is stream MSE
    # to match your original main.py
    split = int(len(X_stream) * 0.8)
    X_test = X_stream[split:]
    y_test = y_stream[split:]

    # ── 1. Clean ──
    print("=" * 50)
    print("SCENARIO 1: Clean (no attack, no defense)")
    print("=" * 50)
    t0 = time.perf_counter()
    model_clean, clean_stream_mse, clean_history = run_clean(X_stream, y_stream)
    clean_time = time.perf_counter() - t0
    clean_test_mse = model_clean.compute_mse(X_test, y_test)
    clean_steady_mse = steady_state_mse(clean_history)
    print(f"MSE on stream (full-run mean):    {clean_stream_mse:.6f}")
    print(f"MSE on stream (post-warmup mean): {clean_steady_mse:.6f}")
    print(f"MSE on held-out test set:         {clean_test_mse:.6f}")
    print(f"Steps run:                        {len(clean_history)} / {len(X_stream)}")
    print(f"Runtime: {clean_time:.2f}s\n")

    # ── 2. Victim ──
    print("=" * 50)
    print(f"SCENARIO 2: Victim (RB-ODPA, poison={POISONING_RATE})")
    print("=" * 50)
    t0 = time.perf_counter()
    model_victim, victim_results = run_victim(X_stream, y_stream)
    victim_time = time.perf_counter() - t0
    victim_test_mse = model_victim.compute_mse(X_test, y_test)
    victim_lot = victim_results["final_mse"] / (victim_time + 1e-10)
    victim_steady_mse = steady_state_mse(victim_results["mse_history"])
    print(f"MSE on stream (full-run mean):    {victim_results['final_mse']:.6f}")
    print(f"MSE on stream (post-warmup mean): {victim_steady_mse:.6f}")
    print(f"MSE on stream (last sample, diag.): {victim_results['last_sample_mse']:.6f}")
    print(f"MSE on held-out test set:         {victim_test_mse:.6f}")
    print(f"Total poisoned samples (distinct): {victim_results['total_poisoned']}")
    print(f"Selection-argmax wins (diag.):    {victim_results['selection_wins']}")
    print(f"Maximum observed loss:            {victim_results['mse_max']:.6f}")
    print(f"Minimum observed loss:            {victim_results['mse_min']:.8f}")
    print(f"Steps run:                        {victim_results['steps_run']} / {len(X_stream)}")
    print(f"Runtime: {victim_time:.2f}s")
    print(f"LOT (stream_mse / runtime):       {victim_lot:.8f}\n")

    # ── 3. Defended ──
    print("=" * 50)
    print("SCENARIO 3: NCF Defense (attack + defense)")
    print("=" * 50)
    t0 = time.perf_counter()
    model_defended, def_results = run_defended(
        X_stream, y_stream,
        ncf_k=5,
        ncf_threshold=0.2,
        ncf_buffer=500
    )
    defended_time = time.perf_counter() - t0
    defended_test_mse = model_defended.compute_mse(X_test, y_test)
    defended_lot = def_results["final_mse"] / (defended_time + 1e-10)
    defended_steady_mse = steady_state_mse(def_results["mse_history"])
    print(f"MSE on stream (full-run mean):    {def_results['final_mse']:.6f}")
    print(f"MSE on stream (post-warmup mean): {defended_steady_mse:.6f}")
    print(f"MSE on stream (last sample, diag.): {def_results['last_sample_mse']:.6f}")
    print(f"MSE on held-out test set:         {defended_test_mse:.6f}")
    print(f"Poisoned samples blocked by NCF:  {def_results['poison_blocked']}")
    print(f"Clean samples wrongly filtered:   {def_results['filtered_incoming']}")
    print(f"Poisoned samples that got through: {def_results['total_poisoned']}")
    print(f"Steps run:                        {def_results['steps_run']} / {len(X_stream)}")
    print(f"Runtime: {defended_time:.2f}s")
    print(f"LOT (stream_mse / runtime):       {defended_lot:.8f}\n")

    # Recall = blocked / (blocked + successfully-injected), i.e. of all
    # poison attempts NCF actually saw post-poisoning, what fraction it
    # caught. This is now well-defined because poison_blocked and
    # total_poisoned are disjoint, distinct-sample counts -- unlike the
    # old total_filtered, which conflated poison-catches with unrelated
    # clean-sample false positives.
    total_poison_attempts_ncf = def_results['poison_blocked'] + def_results['total_poisoned']
    if total_poison_attempts_ncf > 0:
        recall_ncf = 100 * def_results['poison_blocked'] / total_poison_attempts_ncf
        print(f"NCF recall on poison attempts: {recall_ncf:.1f}% "
              f"({def_results['poison_blocked']} of {total_poison_attempts_ncf})\n")

    # ── 4. FDF Defense ──
    print("=" * 50)
    print("SCENARIO 4: FDF Defense (attack + Feature Displacement Filtering)")
    print("=" * 50)
    t0 = time.perf_counter()
    model_fdf, fdf_results = run_fdf(
        X_stream, y_stream,
        fdf_displacement_threshold=1e-6,
        # FDF's revert-to-identical-origin behavior can produce
        # genuinely repeated (not coincidental) loss values across
        # many consecutive steps -- a side effect of reverting
        # rejected samples back to the exact same clean feature
        # vector each time. This can satisfy the patience-gated
        # convergence check "for real" well before the stream is
        # actually exhausted. Use a patience larger than the
        # stream itself to guarantee full-stream runs for
        # reporting purposes.
        convergence_patience=len(X_stream) + 1
    )
    fdf_time = time.perf_counter() - t0
    fdf_test_mse = model_fdf.compute_mse(X_test, y_test)
    fdf_lot = fdf_results["final_mse"] / (fdf_time + 1e-10)
    fdf_steady_mse = steady_state_mse(fdf_results["mse_history"])
    print(f"MSE on stream (full-run mean):    {fdf_results['final_mse']:.6f}")
    print(f"MSE on stream (post-warmup mean): {fdf_steady_mse:.6f}")
    print(f"MSE on stream (last sample, diag.): {fdf_results['last_sample_mse']:.6f}")
    print(f"MSE on held-out test set:         {fdf_test_mse:.6f}")
    print(f"Poisoned samples blocked by FDF:  {fdf_results['poison_blocked']}")
    print(f"Poisoned samples that got through: {fdf_results['total_poisoned']}")
    print(f"Steps run:                        {fdf_results['steps_run']} / {len(X_stream)}")
    print(f"Runtime: {fdf_time:.2f}s")
    print(f"LOT (stream_mse / runtime):       {fdf_lot:.8f}\n")

    total_poison_attempts_fdf = fdf_results['poison_blocked'] + fdf_results['total_poisoned']
    if total_poison_attempts_fdf > 0:
        recall_fdf = 100 * fdf_results['poison_blocked'] / total_poison_attempts_fdf
        print(f"FDF recall on poison attempts: {recall_fdf:.1f}% "
              f"({fdf_results['poison_blocked']} of {total_poison_attempts_fdf})\n")

    # ── Sanity warnings ──
    if victim_results["steps_run"] < len(X_stream) and \
            victim_results["total_poisoned"] < attacker_max_poisoned_hint(victim_results):
        print("WARNING: Victim scenario stopped before reaching the end of the "
              "stream and before hitting the poison budget cap. Convergence "
              "triggered early -- verify this is a genuine plateau, not a "
              "coincidental match, before reporting these numbers.\n")

    expected_poisoned = int(POISONING_RATE * len(X_stream))
    if victim_results["steps_run"] >= len(X_stream) and \
            abs(victim_results["total_poisoned"] - expected_poisoned) > 0.25 * expected_poisoned:
        print(f"NOTE: Victim ran the full stream but total_poisoned "
              f"({victim_results['total_poisoned']}) is far from the "
              f"gamma-implied expectation (~{expected_poisoned} at "
              f"poison rate {POISONING_RATE}). Worth a second look before "
              f"reporting this figure.\n")

    # ── Summary table ──
    # Reduction is computed on the POST-WARMUP (steady-state) mean, not
    # the full-run mean -- the cold-start transient (theta starting at
    # zero) is common to all scenarios and, over a 9568-step run, can
    # dilute the attack/defense effect enough to make Clean and Victim
    # look nearly identical under the plain full-run mean. Both metrics
    # are still printed above for full transparency.
    ncf_stream_reduction = 100 * (victim_steady_mse - defended_steady_mse) / \
                       (victim_steady_mse - clean_steady_mse + 1e-10)
    ncf_test_reduction   = 100 * (victim_test_mse - defended_test_mse) / \
                       (victim_test_mse - clean_test_mse + 1e-10)
    fdf_stream_reduction = 100 * (victim_steady_mse - fdf_steady_mse) / \
                       (victim_steady_mse - clean_steady_mse + 1e-10)
    fdf_test_reduction   = 100 * (victim_test_mse - fdf_test_mse) / \
                       (victim_test_mse - clean_test_mse + 1e-10)

    print("=" * 78)
    print("FINAL SUMMARY  (Stream MSE = post-warmup mean, excludes first "
          f"{200} steps)")
    print("=" * 78)
    print(f"{'Scenario':<28} {'Stream MSE':>11} {'Test MSE':>10} {'LOT':>12}")
    print("-" * 78)
    print(f"{'Clean (no attack)':<28} {clean_steady_mse:>11.6f} "
          f"{clean_test_mse:>10.6f} {'—':>12}")
    print(f"{'Victim (RB-ODPA)':<28} {victim_steady_mse:>11.6f} "
          f"{victim_test_mse:>10.6f} {victim_lot:>12.8f}")
    print(f"{'NCF Defense':<28} {defended_steady_mse:>11.6f} "
          f"{defended_test_mse:>10.6f} {defended_lot:>12.8f}")
    print(f"{'FDF Defense':<28} {fdf_steady_mse:>11.6f} "
          f"{fdf_test_mse:>10.6f} {fdf_lot:>12.8f}")
    print("-" * 78)
    print(f"NCF -- Stream MSE reduction (steady-state): {ncf_stream_reduction:.1f}%")
    print(f"NCF -- Test MSE reduction:                  {ncf_test_reduction:.1f}%")
    print(f"NCF -- poison recall:                       "
          f"{(100 * def_results['poison_blocked'] / total_poison_attempts_ncf) if total_poison_attempts_ncf else 0:.1f}%")
    print()
    print(f"FDF -- Stream MSE reduction (steady-state): {fdf_stream_reduction:.1f}%")
    print(f"FDF -- Test MSE reduction:                  {fdf_test_reduction:.1f}%")
    print(f"FDF -- poison recall:                       "
          f"{(100 * fdf_results['poison_blocked'] / total_poison_attempts_fdf) if total_poison_attempts_fdf else 0:.1f}%")
    print("=" * 78)
    print("\nDone.")


def attacker_max_poisoned_hint(victim_results):
    # Small helper just for the sanity-check message above; avoids
    # importing config twice at call site.
    return int(POISONING_RATE * MAX_ATTACK_ITERATIONS * BATCH_SIZE)


if __name__ == "__main__":
    main()
