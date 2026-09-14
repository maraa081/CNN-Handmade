---
language:
  - en
license: mit
library_name: pytorch
tags:
  - image-classification
  - adversarial-robustness
  - adversarial-training
  - pgd
  - autoattack
  - mnist
  - kmnist
  - curriculum-learning
datasets:
  - mnist
  - kmnist
pipeline_tag: image-classification
---

# MNIST/KMNIST CNNs: the ORDER of the adversarial budget is what hardens the network

Small convolutional networks (421,642 parameters) trained against PGD with
different **internal attack budgets**, released together with the evaluation code
that produced every number below. The point of this release is not a new
state-of-the-art: it is a **controlled, reproducible finding** about how
adversarial training should schedule its own attack.

**Headline result.** At equal cost and equal final internal budget, what changes
the robustness is the **order** in which the budget is visited. On MNIST, at
eps=0.30, judged by AutoAttack (10,000 images):

| recipe (internal attack) | clean | robust (AutoAttack) |
|---|---|---|
| adaptive budget per batch, difficulty target 0.5 (`bande_cible50`) | 98.85% | **91.25%** |
| deterministic schedule `0.2 -> 2 eps` (`A6`) | 99.17% | **82.40%** |
| deterministic schedule `0.2 -> 1 eps` (`A8`) | 99.27% | **78.25%** |
| **reversed** schedule `2 -> 0.2 eps` (`a9_plan_inverse`) | 99.50% | **50.35%** |
| constant 2 eps, hard start (`abl_a`) | 99.53% | **50.71%** |

The two schedules `0.2 -> 2` and `2 -> 0.2` visit **exactly the same budgets**,
the same number of steps, for the same cost. They differ only in order, and they
are **32.1 points** apart (house suite: 22 points). Reversed on KMNIST (kana),
the gap is **49.5 points** (house: 41). AutoAttack - a standardised judge,
stronger than our suite, not an "unbiased" one - amplifies the effect: the
objection "maybe the gap is an artifact of your optimistic suite" is returned
to sender.

Note on prior work: an increasing budget schedule is not new, it is the principle
of *Curriculum Adversarial Training* (Cai et al., IJCAI 2018). The contribution
claimed here is the controlled comparison - the same set of budgets in reverse
order, at identical cost, on the same architecture - together with the log-based
diagnosis and the KMNIST replication.

For reference, on MNIST at eps=0.30 the literature reports 96.01% (Madry et al.,
PGD-40) and 95.60% (TRADES). Our best is ~5 points below; our constant-budget
`abl_a`, in the **same** architecture, is 45 points below.

## Files

| file | what it is | clean | robust (AutoAttack, eps=0.30) |
|---|---|---|---|
| `bande_cible50.pt` | best model: adaptive per-batch budget, difficulty target 0.5 | 98.85% | **91.25%** |
| `a6_plan_budget.pt` | cheap and strong: schedule `0.2 -> 2 eps` (11 min of training) | 99.17% | **82.40%** |
| `a8_plan_0p2_1.pt` | schedule `0.2 -> 1 eps` | 99.27% | **78.25%** |
| `abl_a_eps10.pt` | reference baseline: constant 2 eps, hard start | 99.53% | 50.71% |
| `standard.pt` | undefended model, same architecture (for comparison) | 98.6% | 1.8% (FGSM, eps=0.30) |

KMNIST twins (same recipes, different dataset) are available on request; they
are the "the recipe transfers, the level does not" half of the result: `0.2 -> 2`
drops from 82.40% to 58.53%, `0.2 -> 1` from 78.25% to 51.03%.

## Architecture

Identical to the hand-written NumPy reference implementation in the repository:

```
Conv2d(1 -> 32, k=3, pad=1) -> ReLU -> MaxPool2d(2)
Conv2d(32 -> 64, k=3, pad=1) -> ReLU -> MaxPool2d(2)
Flatten(3136) -> Linear(3136 -> 128) -> ReLU -> Linear(128 -> 10)
```

421,642 parameters, 28x28 grayscale input, 10 classes. Training recipe: 60,000
MNIST training images, 120 epochs, data augmentation, PGD-20 with step
`alpha = eps/10`, `eps = 0.30`, seed 42. The only thing that changes between the
models above is the **schedule of the internal attack budget** (constant,
increasing, decreasing, or adaptive per batch). Exact commands:
`adversarial/README.md`, section "Le catalogue des runs".

## Usage

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

class CNN(nn.Module):
    """Same architecture as src/model.py of the repository."""
    def __init__(self, num_classes=10, large=False):
        super().__init__()
        c1, c2, f = (64, 128, 256) if large else (32, 64, 128)
        self.conv1 = nn.Conv2d(1, c1, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c2 * 7 * 7, f)
        self.fc2 = nn.Linear(f, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

chemin = hf_hub_download("<hf-user>/<hf-repo>", "bande_cible50.pt")
ck = torch.load(chemin, map_location="cpu", weights_only=False)
etat = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

modele = CNN()
modele.load_state_dict(etat)
modele.eval()

# x: tensor (B, 1, 28, 28) normalized the same way as training (mean 0.1307, std 0.3081)
# y = modele(x)
```

An evaluation suite (`FGSM`, `PGD`, `APGD-CE`, `APGD-DLR`, `Square`, `NES`,
`CW-L2`, `Boundary`) and the AutoAttack wrapper are in the repository
(`adversarial/torch/`); the numbers above are produced by them, not copied by
hand.

## How the numbers were measured

Six measurement rules are written in `adversarial/README.md`. The three that
matter for reading this card:

1. **The advertised number is AutoAttack** (`standard` version, 10,000 images,
   eps=0.30, L-infinity), never a single attack of our own.
2. **Our house suite (500 images) is optimistic, and we publish the bias**: from
   -2.4 to -7.4 points for recipes starting low, -12.5 to -15.5 points for
   recipes starting high. The bias always goes the same way (optimistic) and it
   *separates* the two families, which is an argument in favour of the finding,
   not against it.
3. **A gradient-based radius is never announced as a robustness measure**; it is
   used to *sort* and to *explain*, AutoAttack remains the judge.

## Limitations (stated, not hidden)

- **Official numbers are single-seed.** A 4-seed plan on the flagship pair gives
  means 86.9% (sd 1.4) vs 61.2% (sd 3.1) on the house suite: the gap between
  means is 25.7 points and the *smallest* seed-to-seed gap is 21.8 points = 7x
  the largest standard deviation. The effect is much larger than the noise, but
  the official figures themselves are still one seed.
- **KMNIST does not reach the MNIST level** (-23.9 points at identical recipe):
  the recipe transfers, the level does not. The peak of the "budget bell" moves
  with the difficulty of the dataset.
- **Our TRADES variant does not reproduce the reference** (22.01% / 25.18% vs
  95.60% in the paper); two implementation differences are identified in the
  code. It is published as a limitation, and is **not** used to compare
  methods.
- **Small architecture, single dataset pair, eps=0.30 L-infinity only.** No
  ensemble, no training against Square, nothing above 1.7M parameters, no
  cross-dataset transfer: those are explicit non-goals.
- The 91.25% model is the expensive one (adaptive per-batch feedback loop);
  the 82.40% model is 11 minutes of training on a laptop GPU.

## Links

- Repository (code, attacks, logs, article): https://github.com/maraa081/CNN-Handmade
- Article, French: `adversarial/article/ARTICLE-fr.md`
- Article, English: `adversarial/article/ARTICLE-en.md`
- Lab notebook (every run, including the failed ones): `adversarial/memoire.md`

## Cite

```bibtex
@misc{lazurka2026cnnhandmade,
  title  = {The order of the adversarial budget is what hardens the network},
  author = {Lazurka, Maxime},
  year   = {2026},
  note   = {Code and article: https://github.com/maraa081/CNN-Handmade}
}
```
