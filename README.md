

# RB-ODPA: Online Data Poisoning Attack

Implementation of **RB-ODPA (Rehearsal-Buffer Online Data Poisoning Attack)** against an edge-device linear regression model using the 9,568-sample Combined Cycle Power Plant (CCPP) dataset.

This project reproduces the online poisoning attack, audits its evaluation methodology, and investigates defensive approaches against feature-based data poisoning.

## Overview

- Reimplemented RB-ODPA on the CCPP data stream.
- Investigated discrepancies between reproduced and published results.
- Identified the impact of omitting the intercept in linear regression.
- Evaluated clean training, poisoned training, and defensive filtering.
- Implemented and evaluated two defences: NCF and Feature Displacement Filtering (FDF).

## Key Findings

### Evaluation and reproduction

- A like-for-like evaluation showed an attack-induced stream MSE increase of approximately 4 \[specify whether 4×\].
- Linear regression without an intercept had a best achievable MSE of approximately **0.017**, compared with **0.0036** with an intercept.
- The published unpoisoned MSE was approximately **0.0035**.

### Defence results

| Defence | Poison detection recall |
| --- | --- |
| NCF (Label-consistency filtering) | 0% |
| FDF (Feature Displacement Filtering) | 52.7% |

NCF failed to detect poisoned samples in the corrected evaluation. Its previously reported 86% recall was traced to an incorrect counter that included rejected clean samples.

FDF detects feature displacement, directly targeting the poisoning mechanism used by RB-ODPA.

**Note:** MSE reduction percentages are not finalized because the victim and defended scenarios currently use different poisoning-budget policies.

## Installation

Clone the repository:

bash

```bash
git clone https://github.com/esho-he-boisakhh/Online-DPA/

```

Install the dependencies:

bash

```bash
pip install -r requirements.txt
```

The project dependencies are listed in `requirements.txt`.

## Running the Experiments

Run the main poisoning attack:

bash

```bash
python main.py
```

Run the defended experiments (Clean, Victim, NCF and FDF):

bash

```bash
python main_defended.py
```

## Implementation Details

- `main.py` – Main RB-ODPA experiment.
- `main_defended.py` – Defence experiments and result comparisons.
- `online_dpa.py` – Online poisoning attack implementation.
- `linear_regression.py` – Linear regression model.
- `online_trainer.py` – Online training and Jacobian bookkeeping.
- `utils.py` – Supporting utilities.
- `requirements.txt` – Python dependencies.

## Notes

- Experiments use the 9,568-sample CCPP dataset.
- Convergence-patience checks prevent premature termination.
- Poisoning counts track distinct poisoned samples.
- FDF uses a default displacement threshold of `1e-6`.
- MSE reduction comparisons should not be reported until all scenarios use a consistent poisoning-budget policy.
