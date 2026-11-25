============================================================
  RB-ODPA + NCF Defense -- FIXED BUILD (v2)
============================================================

WHAT'S FIXED (in order discovered)
-----------------------------------
1. Convergence check (online_dpa.py, main_defended.py)
   Fixed to require `convergence_patience` consecutive stable
   steps before an early break, instead of a single coincidental
   floating-point match. Runs now reliably process the full
   9568-sample stream (steps_run == 9568) instead of stopping
   partway through non-reproducibly.

2. Jacobian bookkeeping (linear_regression.py, online_trainer.py,
   utils.py) -- THE BIG ONE
   jacobian_history was a flat list only appended to on
   batch_size==1 steps, but the attack loop trains on combined
   multi-sample batches almost the entire run. This starved the
   recursive-gradient chain almost immediately after warm-up,
   causing most poisoning attempts to silently fail (swallowed
   by try/except) or use misaligned Jacobians.

   Fix: jacobian_history is now a dict keyed by TRUE stream time
   index, populated on every step using the correct Jacobian for
   whatever update actually happened (single-sample formula, or a
   new batch-mean formula for combined-batch steps). Verified:
   total_poisoned now reaches ~1000 (the gamma-implied budget) on
   the full stream, matching your original Table 4.1 figures.

3. Poison-count semantics (online_dpa.py, main_defended.py)
   total_poisoned now counts DISTINCT stream samples that were
   actually poisoned (tracked by identity), not the old
   "selection-argmax win" counter, which measured something
   unrelated and drastically undercounted real poisoning
   activity.

4. Metric stability (online_dpa.py, main_defended.py)
   Canonical "Stream MSE" is the MEAN over the run, not a single
   last-sample value (too noisy -- was producing nonsensical
   figures like >100% "reductions"). A POST-WARMUP mean
   (excluding the first 200 steps) is also computed and used for
   the Clean/Victim/NCF comparison, since the cold-start
   transient (theta starting at zero) otherwise dilutes the
   attack/defense effect across a 9568-step run. Both are printed
   so you can see the difference.

5. *** CRITICAL FINDING: NCF filter-counter conflation ***
   The old total_filtered counter summed TWO unrelated events:
   (a) clean incoming samples rejected by NCF before poisoning
       was even attempted (false positives), and
   (b) poisoned buffer samples caught by NCF's re-check (actual
       defense successes).
   Splitting these apart reveals: NCF recall on poison attempts
   is 0% in this implementation -- every poisoned sample gets
   through, and the "100 filtered" figure your report cited was
   entirely false-positive rejections of CLEAN samples, not
   caught poison. The modest stream-MSE improvement you'll still
   see for the NCF scenario is NOT evidence of working detection;
   it's most likely an incidental side effect of excluding 100
   samples (poisoned or not) from training.

   This means the "86% recall" figure in your report cannot be
   salvaged by recomputation -- it needs to be either retracted
   and reported as a negative result (NCF fails to detect
   RB-ODPA's poison, analogous to the three failed CARA defenses
   already in Chapter 6), or the NCF trigger logic needs a real
   redesign before recall can be claimed. See main_defended.py's
   printed "NCF recall on poison attempts" line and the
   poison_blocked / total_poisoned / filtered_incoming fields.

HOW TO RUN
----------
1. pip install numpy pandas openpyxl scikit-learn
2. python main.py              -> single attack run (Table 4.1 layout)
3. python main_defended.py     -> Clean / Victim / NCF comparison (Table 5.1 layout)

WHAT TO CHECK IN THE OUTPUT
----------------------------
- steps_run should be 9568/9568 for all three scenarios (NCF's
  will be slightly less -- filtered_incoming samples don't get a
  training step, by design)
- Victim's total_poisoned should land at/near 1000 (the hard
  poison-budget cap at gamma=0.1, max_iter=10000)
- "NCF recall on poison attempts" -- currently 0%, see finding #5
  above before reporting any recall figure

TUNING
------
CONVERGENCE_PATIENCE / CONVERGENCE_REL_TOL (main_defended.py,
also constructor args on OnlineDPA): control how strict the
convergence-based early stop is. Defaults force effectively
full-stream runs, which is what you want for reproducible
reporting.

WARMUP_STEPS (main_defended.py, steady_state_mse function):
how many initial steps to exclude from the "post-warmup mean"
metric. Default 200; adjust if your model converges faster or
slower.

NCF params (ncf_k, ncf_threshold, ncf_buffer) are unchanged --
set them in the run_defended() call inside main_defended.py's
main(). Given finding #5, you may want to experiment with
ncf_threshold specifically to see if any setting gives NCF real
recall on this attack, before concluding it's a genuine negative
result.
============================================================

============================================================
  ADDENDUM: FDF (Feature Displacement Filtering) -- v3
============================================================

NEW: Scenario 4, a genuine defense targeted at RB-ODPA's real
mechanism (feature-vector perturbation via projected gradient
ascent, label untouched). See FDFFilter class and run_fdf() in
main_defended.py for full documentation in code comments.

VERIFIED RESULT: FDF achieves 52.7% recall on poison attempts,
vs NCF's 0% (see main NCF findings above). This is a real,
reproducible improvement -- confirmed by direct instrumentation
of the poisoning displacement values, not just the end-to-end
recall number.

displacement_threshold default is 1e-6 (normalized [0,1]^4
feature space). DO NOT set this much lower (e.g. 1e-9) --
measured to cause a severe runtime blowup (samples get reverted
so often that the poisoning optimizer's inner loop stops
converging early and burns its full 500-iteration budget on
nearly every pass). 1e-6 is a deliberate, tested trade-off point,
not an arbitrary default.

*** KNOWN LIMITATION -- READ BEFORE CITING MSE REDUCTION % ***
The Victim scenario still runs under the original hard poison-
budget cap (max_poisoned = gamma * max_iter * batch_size = 1000).
The defended scenarios (NCF, FDF) had this cap REMOVED, because
under an active defense the same sample identity can succeed and
later get caught repeatedly across many passes (revert-and-retry
under FDF specifically), which inflated the "successful poison"
count far faster than genuine attacker progress and was causing
early, artificial stream truncation.

This means Victim's "1000 poisoned" and FDF's "1148 got through"
are NOT directly comparable counts (different stopping/budget
policies) -- so the Stream-MSE-reduction and Test-MSE-reduction
PERCENTAGES currently printed should NOT be reported as-is. The
recall figure itself (poison_blocked / (poison_blocked +
total_poisoned), computed within a single scenario) IS
self-consistent and safe to report.

FIX NEEDED before finalizing MSE-reduction numbers: apply the
SAME budget policy to all four scenarios -- either remove the
hard cap from Victim too (so all scenarios run under identical,
uncapped rules), or reintroduce a cap for the defended scenarios
that's conceptually equivalent (e.g. cap on ATTEMPTS rather than
DISTINCT SUCCESSES). This is a scoped, mechanical fix -- ask if
you want it done before your next report revision.
============================================================
