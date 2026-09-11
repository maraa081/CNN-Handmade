# CNN Handmade

**English** | [Français](README.md)

> **TL;DR**
> - No TensorFlow, no PyTorch, no Keras: Python and NumPy only. Every layer is written by hand (im2col forward, backward, update).
> - Clean MNIST: **98.6%** test accuracy; the optimizers (SGD, Momentum, Adam) are hand-made too.
> - Adversarial attacks, hand-made as well: FGSM drops the full model to **1.8%**, PGD to **0.0%** at eps=0.30.
> - Hardened model (PyTorch, 60000 images, PGD-5): **98.8% clean**, **65.4% under PGD-20** at eps=0.30.
> - Hardening campaign A/B/C: run A is the reference, augmentation (B) hurts at equal epoch budget, TRADES (C) still needs another go.
> - Two engines, same maths: hand-made NumPy and PyTorch, ~9x faster on CPU, interchangeable `.npz` weights.
> - Next step: adaptive attacks (BPDA/EOT), Carlini-Wagner, black-box (ZOO/NES, Boundary/HSJA), randomized smoothing.

**A convolutional neural network for recognising handwritten digits (MNIST), built by hand, from A to Z.**

No TensorFlow, no PyTorch, no Keras. Just Python, NumPy, and me. 

## Why?

Understanding every building block of deep learning by coding it yourself — im2col, backpropagation, gradient descent... rather than calling a magic API.

## What is implemented

| Module | Status |
|---|---|
| **MNISTLoader** (raw IDX files) | OK |
| **Preprocessing** (normalisation, one-hot, DataLoader) | OK |
| **Conv2D** (im2col forward, backward, update) | OK |
| **MaxPool2D** (forward + backward with indices) | OK |
| **ReLU** (forward + backward) | OK |
| **Flatten** (forward + backward) | OK |
| **Dropout** (regularisation by random deactivation) | OK |
| **Dense / Fully Connected** (forward + backward + update) | OK |
| **Softmax** (forward + backward) | OK |
| **CrossEntropyLoss** (forward + backward + accuracy) | OK |
| **Training loop** (CNN.train + train/eval mode) | OK |
| **Evaluation** (accuracy on the test set) | OK |
| **Weight saving / loading** | OK |
| **Predict** (classify a loaded image) | OK |
| **Training plots** (loss + accuracy) | OK |
| **SGD optimizer** (vanilla + weight_decay) | OK |
| **Momentum optimizer** (SGD + momentum + weight_decay) | OK |
| **Adam optimizer** (adaptive lr + momentum + weight_decay) | OK |
| **Experimentation framework** (optimizer comparison) | OK |
| **Adversarial attacks** (FGSM, PGD, targeted, transfer) | OK |
| **Defences** (FGSM adversarial training, hardened PGD version, feature squeezing) | OK |

## Architecture

```
Input : (N, 1, 28, 28)
    |
    |-- Conv2D   (1 -> 32,  kernel=3, pad=1)     ->  (N, 32, 28, 28)
    |-- ReLU
    |-- MaxPool2D  (2×2, stride=2)               ->  (N, 32, 14, 14)
    |
    |-- Conv2D   (32 -> 64, kernel=3, pad=1)      ->  (N, 64, 14, 14)
    |-- ReLU
    |-- MaxPool2D  (2×2, stride=2)               ->  (N, 64, 7, 7)
    |
    |-- Flatten                                  ->  (N, 3136)
    |-- Dense   (3136 -> 128)
    |-- ReLU
    |-- Dropout(p=0.5)          (optional)
    |-- Dense   (128 -> 10)
    `-- Softmax                                  ->  (N, 10)
```

## Project structure

```
CNN-Handmade/
|-- README.md
|-- requirements.txt
|-- push.sh                        <- quick git push
|-- src/                           <- core of the code (CNN from scratch)
|   |-- __init__.py
|   |-- data.py        — MNISTLoader, EMNISTLoader, preprocessing, DataLoader
|   |-- layers.py      — im2col/col2im, Conv2D, MaxPool2D, ReLU, Flatten, Dropout, Dense
|   |-- losses.py      — Softmax, CrossEntropyLoss
|   |-- optimizers.py  — SGD, Momentum, Adam
|   |-- model.py       — CNN (swappable optimizer, train/eval mode)
|   |-- tune_cnn.py    — interactive settings (baseline)
|   `-- cnn.py         — main demo + training script
|-- scripts/                      <- user scripts
|   |-- predict.py        — load the model and classify
|   |-- train_emnist.py   — train on EMNIST letters
|   |-- download_emnist.py— install the EMNIST data
|   `-- voir_emnist.py    — visualise the letters (Spyder/IPython)
|-- models/                      <- trained weights (.npz)
|   |-- model_weights.npz          — quick (2000 img, 3 epochs)
|   |-- model_weights_full.npz     — full (60000 img)
|   `-- max_config_weights.npz     — Adam + Dropout + L2
|-- results/                     <- generated plots (PNG/CSV)
|-- adversarial/                 <- AI security: attacks & defence
|   |-- README.md                     - how-to + key results
|   |-- memoire.md                    - experiment logbook
|   |-- attacks.md                    - attack theory (FGSM, PGD, transfer)
|   |-- defenses.md                   - defence theory + methodology
|   |-- results/                      - curves and images (PNG) + logs/ (gitignored)
|   |-- scripts/                      - hand-made implementation (NumPy)
|   |   |-- fgsm.py / pgd.py / transfer.py     - attacks
|   |   |-- defend.py / harden.py / harden2.py - defences
|   |   |-- augment.py                         - data augmentation
|   |   |-- eval_defended.py                   - evaluation without retraining
|   |   |-- bpda_eot.py                        - adaptive attacks (BPDA + EOT)
|   |   `-- campagne.sh                        - the 3 hardened recipes in series
|   `-- torch/                        - PyTorch path (autograd, GPU)
|       |-- modele.py / attaques.py / entrainement.py  - same maths, another engine
|       `-- harden_torch.py                            - entry point (same options)
|-- docs/
|   |-- data-flow.md
|   `-- memoire-projet.md          <- project logbook
|-- data/
|   `-- (MNIST/EMNIST .ubyte files)
|-- traces/
|   `-- forward_trace.py
`-- experiments/                    <- optimizer benchmarks
    |-- baseline/  momentum/  adam/  dropout/  l2/
    |-- max_config.py          —  Adam + Dropout + L2 combined
    `-- compare_all.py         — run all the optimizers at once
```

## Usage

### 1. Install the dependencies

```bash
pip install numpy matplotlib
```

Optional, for the PyTorch path (faster training + GPU):

```bash
pip install torch          # CPU
# or, for an AMD GPU: see adversarial/torch/README.md
```

> [warn] PyTorch does not support every version of Python. Python 3.10 to
> 3.12 is the safest (3.14 is not supported).

Complete recipe if your `python3` is too recent (Windows/Git Bash example):

```bash
py install 3.12
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy
```

### 2. Run the tests + quick training

```bash
python src/cnn.py
```

- Tests every layer one by one (forward, backward, gradient check)
- Trains on **2000 images** (3 epochs)
- Saves the weights to `models/model_weights.npz`
- Generates the plot `training_result.png`

### 3. Full training (recommended)

```bash
python src/cnn.py --full
```

- Trains on **all 60000** MNIST images (10 epochs, ~15-20 min)
- Saves the weights to `models/model_weights_full.npz`
- Generates `training_result_full.png`

Additional options:
```bash
python src/cnn.py --full --epochs 15      # 15 epochs instead of 10
python src/cnn.py --train-only            # skips the tests, trains straight away
```

### 4. Predict without retraining

```bash
python scripts/predict.py                          # 10 predictions -> results/predictions.png
python scripts/predict.py --all                    # accuracy on the 10000 test images
python scripts/predict.py --weights models/model_weights_full.npz   # choose the weights
python scripts/predict.py --interactive            # step-by-step mode with display
```

## Interactive tuning

```bash
python src/tune_cnn.py
```

Tunable parameters:
- `LEARNING_RATE` (0.1, 0.01, 0.001…)
- `BATCH_SIZE` (32, 64, 128…)
- `EPOCHS` (5, 10, 20…)
- `DATA_LIMIT` (2000, 5000, None for everything)
- Network architecture

Result saved to `tune_result.png`.

## EMNIST — Letters (26 classes)

The same from-scratch CNN, but to recognise **handwritten letters a-z** instead of digits.

```bash
# Check the orientation of the images (samples -> results/emnist_samples.png)
python3 scripts/train_emnist.py --samples

# Quick training (5000 images, 3 epochs, ~4 min)
python3 scripts/train_emnist.py

# Full training (124800 images, 10 epochs)
python3 scripts/train_emnist.py --full
```

**Data:** [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) — same IDX format as MNIST. A lightweight zip of the letters (~36 MB) is included in the repo; the script installs them on its own:

```bash
python3 scripts/download_emnist.py
```

If the local zip is not there, it downloads automatically from the NIST site (~561 MB, slower).

**What changes vs MNIST:**
- `EMNISTLoader` in `src/data.py` (labels 1-26 -> 0-25, rotated images put back upright)
- `Dense(128 -> 26)` instead of `Dense(128 -> 10)`
- `preprocess_pipeline(..., num_classes=26)` for the one-hot

**Quick result** (5000 images, 3 epochs): **43.2% test** (chance = 3.8%). Full training targets ~90%.

> [warn] **Pitfall**: the EMNIST test set is sorted by class — sample randomly to evaluate, never `x_test[:N]`.

## AI security — attacks and defence (adversarial)

A model at 98.6% can drop to 0% because of noise invisible to the eye.
This is the "AI security" part of the project: attacking my own CNN, then
trying to defend it. Everything is done by hand, without an attack
library.

```bash
# FGSM attack on the full MNIST model
python3 adversarial/scripts/fgsm.py --weights models/model_weights_full.npz --n 1000

# PGD attack (20 iterations, random start) + comparison with FGSM
python3 adversarial/scripts/pgd.py --weights models/model_weights_full.npz --n 500 --compare

# Attack transfer between two models (simulates a black-box attack)
python3 adversarial/scripts/transfer.py --src full --dst max_config --attack pgd

# Defence: PGD adversarial training + feature squeezing, then multi-attack evaluation
python3 adversarial/scripts/harden.py --n-train 5000 --epochs 3

# Adaptive attacks: break the defence (BPDA + EOT)
python3 adversarial/scripts/bpda_eot.py
```

**Key results**

| Experiment | Result |
|---|---|
| FGSM, MNIST full (98.6% clean) | 1.8% at eps=0.30 |
| PGD, MNIST full (98.6% clean) | 0.0% from eps=0.20 |
| PGD, EMNIST full (92.0% clean) | 0.0% from eps=0.20 |
| Transfer full -> classic | 72.3% transfer at eps=0.30 |
| hardened v1 (`harden.py`, 5000 img) | clean 67.2%, 23.8% under FGSM eps=0.30 |
| **campaign A (torch, 60k img, PGD-5)** | **clean 98.8%, 65.4% under PGD eps=0.30** |
| campaign B (A + augmentation) | clean 99.5%, 29.8% under PGD eps=0.30 |
| campaign C (B + TRADES) | clean 96.9%, 2.2% under PGD eps=0.30 |
| Adaptive attacks BPDA+EOT (v1 defence) | gradient masking: the naive attacker leaves 62.5%, BPDA breaks it to 1.5% (eps=0.30) |

> **Reference result (2026-09-10).** The hardened model trained in PyTorch
> on 60000 images (10 epochs, PGD-5, without augmentation) reaches **98.8% clean
> accuracy and 65.4% under PGD-20 at eps=0.30** — against 67.2% and 1.2%
> for the first hardened version.

### The hardening campaign: 3 recipes, a single variable that changes

`adversarial/scripts/campagne.sh` chains 3 identical runs except on one point,
to answer a precise question: *is it the dataset or the method that
matters?*

| | Recipe | Clean | FGSM eps=0.30 | PGD eps=0.30 |
|---|---|---|---|---|
| **A** | reference (60000 img, PGD-5, no augmentation) | **98.8%** | **88.2%** | **65.4%** |
| **B** | A + data augmentation | 99.5% | 54.8% | 29.8% |
| **C** | B + TRADES (beta=2) | 96.9% | 23.4% | 2.2% |

```bash
# The complete campaign (3 runs, automatic resume), in NumPy or PyTorch
./adversarial/scripts/campagne.sh
./adversarial/scripts/campagne.sh --torch

# Short version to check the pipeline before a long run
./adversarial/scripts/campagne.sh --rapide
```

> **Augmentation HURT (run B)** — a counter-intuitive result, but explained:
> the training loss stays stuck at ~0.82 while that of run A goes down to
> 0.24. The augmented model is **under-trained**: each epoch is harder,
> so on an equal epoch budget it falls behind. It gains in clean
> accuracy (99.5%) but loses 35 pts of robustness. Conclusion:
> augmentation only pays off with 2-3x more epochs, and it does not replace
> adversarial training.
>
> **TRADES still needs another go (run C)**: learning rate too low (loss frozen at
> ~1.68) and decay too aggressive. It is a tuning problem, not a
> method problem.

> **Central lesson**: FGSM underestimates the real vulnerability. A model is
> judged against the strongest attack (PGD), not against the simplest one.

**Two engines, same maths.** The `adversarial/scripts/` folder contains
the hand-made implementation (every gradient written by hand: the pedagogical
reference). `adversarial/torch/` replays **exactly the same experiments**
with PyTorch and autograd: same architecture, same attacks, same recipes,
but ~9x faster on CPU and usable on GPU. The weights are
interchangeable between the two (`.npz` format), and the `--parite` mode checks
that both give the same result.

Details and GPU setup: [`adversarial/torch/README.md`](adversarial/torch/README.md).

### What's next

The hardened model holds 65.4% under PGD — but PGD is precisely the attack
against which it was trained. First step done on **2026-09-11**: the adaptive
attacks BPDA + EOT show that the v1 defence (feature squeezing) was only
**gradient masking** (the naive attacker leaves 62.5% to the model at eps=0.30,
BPDA brings it down to 1.5%). Detail in `adversarial/scripts/bpda_eot.py` and
the "Adaptive attacks" section of [`adversarial/README.md`](adversarial/README.md).

| Step | Content | Reference |
|---|---|---|
| 1 | **Adaptive attacks**: BPDA + EOT (**done on 2026-09-11**) | Athalye et al. 2018; Tramèr et al. 2020 |
| 2 | **Carlini-Wagner (CW)**: the reference white-box attack | Carlini & Wagner 2017 |
| 3 | **Black-box**: score-based (ZOO/NES) then decision-based (Boundary/HSJA) | Chen et al. 2017; Brendel & Bethge 2019 |
| 4 | **Certified robustness**: randomized smoothing (guaranteed L2 bound) | Cohen et al. 2019 |

Additional leads already noted: transfer with PGD as source, cross-dataset
transfer (MNIST -> EMNIST), ensemble attack, AutoAttack/RobustBench.

The full detail is in [`adversarial/README.md`](adversarial/README.md):
theory and algorithms in [`attacks.md`](adversarial/attacks.md) and
[`defenses.md`](adversarial/defenses.md), run history in
[`adversarial/memoire.md`](adversarial/memoire.md).

## Experiments — Comparing techniques

Each folder in `experiments/` is an independent test. **Same architecture, same data, only the technique changes.**

### Run a test

```bash
# Optimizers
python experiments/baseline/baseline_sgd.py
python experiments/momentum/momentum.py
python experiments/adam/adam.py

# Regularisation
python experiments/dropout/dropout_sgd.py     # Dropout(p=0.5)
python experiments/l2/l2_sgd.py              # Weight decay L2(0.001)
```

### Planned next experiments

| Experiment | Status |
|---|---|
| Baseline (SGD) | OK |
| Momentum | OK |
| Adam | OK |
| Dropout (regularisation) | OK |
| Weight Decay (L2) | OK |
| Learning Rate Scheduler | [wait] |
| Automatic Grid Search | [wait] |
| Max Config (Adam + Dropout + L2) | OK |
| Data Augmentation | OK (in `adversarial/scripts/augment.py`) |

### Adding a new experiment

1. Create `experiments/mon_opti/mon_opti.py`
2. Import your optimizer from `src/optimizers.py` (or create it there)
3. If you add layers (Dropout, etc.), import them from `src/layers.py`
4. Pass the optimizer to the model: `model = CNN(optimizer=MonOpti(lr=...))`
5. Run it and compare the plots!

## The optimizers — explanations

The optimizers live in `src/optimizers.py`. Each one implements an `update(layers, lr=None)` method.

### SGD — The baseline
```python
θ <- θ - lr · ∇θ
```
Each parameter is updated in the direction opposite to the gradient. Simple, stable, but it can be slow.

### Momentum — With inertia
```python
v <- α · v - lr · ∇θ
θ <- θ + v
```
We accumulate a "velocity" that smooths out the oscillations and speeds up convergence. The `α` coefficient (typically 0.9) controls the inertia.

### Dropout — Random deactivation
```python
# Training: binary mask × scaling
masque ~ Bernoulli(1-p)
sortie = entrée × masque / (1-p)

# Evaluation: pass-through
sortie = entrée
```
Prevents the co-adaptation of neurons. Forces the network to learn redundant representations. Acts as an ensemble of sub-networks. `p=0.5` for the Dense layers, `p=0.2-0.3` for the Conv ones.

### L2 Weight Decay — Penalises large weights
```python
∇θ_effectif = ∇θ + λ · θ
θ <- θ - lr · ∇θ_effectif
```
Adds a quadratic penalty on the weights. Weights that are too large are pulled towards zero. It amounts to looking for simpler solutions. Typical `λ`: 0.0001 ~ 0.001.

### Adam — The champion
```python
m <- β1 · m + (1 - β1) · g         (mean of the gradients)
v <- β2 · v + (1 - β2) · g²        (variance of the gradients)
θ <- θ - lr · m / (√v + ε)
```
Combines momentum with a per-parameter adaptive learning rate. The most robust one — less need to tune the lr. Also supports weight_decay.

## Quick example

```python
from scripts.predict import load_model

model = load_model("models/model_weights_full.npz")
pred = model.predict(mon_image)   # mon_image: (1, 28, 28) normalisée
print(f"Prédiction : {pred}")
```

## What we learned (for what comes next)

### Key results (5 epochs, 2000 images)

| Technique | Test Acc | Time | Verdict |
|---|---|---|---|
| **SGD** (baseline) | 88.4% | 20s | Reference |
| **Momentum** | 94.2% | 21s | +6% for almost nothing |
| **Adam** | 95.5% | 24s |  Best on its own |
| SGD + Dropout | 83.3% | 23s | To be kept for large datasets |
| SGD + L2 | 86.4% | 20s | Same, of little use on MNIST |
| **Max Config** (10 epochs) | **96.0%** | 49s | Everything combined, best long term |

### Takeaways

1. **Adam is enough on MNIST** — 99% on the 60000 images, no need for regularisation
2. **Dropout + L2** -> useful on real problems (overfitting), not on clean MNIST
3. **The best performance/simplicity ratio**: just Adam
4. **Max Config** useful when you scale up epochs or data

### Next leads

- **LR Scheduler** — reduce the lr during training (step decay, cosine annealing)
- **Grid Search** — automatically find the best hyperparameters
- **Data Augmentation** — implemented on the adversarial side: `adversarial/scripts/augment.py`
  (rotation, zoom, translation, impulse noise, cutout, stroke thickness)
- **Full training** (60000 images, 20 epochs) -> aiming for 99%+

---

**#NoFrameworks #FromScratch #MNIST**
