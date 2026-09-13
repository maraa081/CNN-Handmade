# The order in which you present the difficulty is what builds robustness

**A law about the displacement budget of the inner attack: found, replicated on a
second dataset, and passed to the official judge.**

*Work in progress: English translation of `ARTICLE-fr.md` (the French version is
the reference until this file is complete). Sections translated so far: summary,
1, 2. Figures live in `figures/` and are language-neutral (English labels).
Sources of every number: `adversarial/memoire.md` and the JSON files in
`adversarial/results/logs/`.*

---

## Summary

This work starts from a CNN written by hand in NumPy, breaks it, defends it, and
then breaks its own defence. The central result is not a score: it is a **law
about how the difficulty should be presented during adversarial training**.

A model trained against an inner attack with a **decreasing** budget
(`2 -> 0.2 eps`, 11 minutes of compute) reaches **63.8%** worst-case accuracy.
The same model, trained against exactly the same budgets in **increasing** order
(`0.2 -> 2 eps`, same cost, same 11 minutes), reaches **85.8%**. Twenty-two
points of gap, one single variable changed: the **order**.

A seed study (4 seeds per recipe) confirms that the gap is not training noise:
the smallest gap observed between an increasing-order seed and a decreasing-order
seed is **21.8 points**, seven times the dispersion.

And the official judge **amplifies** the effect instead of reducing it. Under
AutoAttack on 10,000 images, the two members of the pair give **82.40%** against
**50.35%**, a **32-point gap** — larger than the one measured by our own attack
suite. It is the first time in this work that an unbiased measurement moves in
the same direction as the biased one, and further.

The result replicates on a second dataset (KMNIST, Japanese cursive kana), where
the official gap between the two orders reaches **49.5 points** (51.03% against
1.49%, same budgets, same cost). The *level*, however, does not transfer: at an
identical recipe, official numbers go from 82.40% (MNIST) to 58.53% (KMNIST),
i.e. **-23.9 points**. The formulation we keep is therefore: *the recipe
transfers, the level does not*.

Three methodological choices structure the rest: every announced figure comes
from **AutoAttack** on 10,000 images; our own attack suite is presented as a
diagnostic tool, with its bias measured (2 to 15.5 points of optimism depending
on the model); and the adoption criterion of every recipe is written **before**
the corresponding run, including when it turns out to be wrong.

---

## 1. Introduction

### 1.1 Why write a neural network by hand

The starting point is an exercise: write a CNN with no framework, in Python and
NumPy only. Every building block is hand-made — `im2col` for the convolution,
backpropagation, the optimizers (SGD, Momentum, Adam), the attack formulas, and
even the confidence bounds of smoothing. The network has 421,642 parameters and
reaches 98.6% on MNIST.

This exercise has a hidden benefit for everything that follows. When we later
attack the network, we know exactly what we are attacking: there is no hidden
library layer, no undocumented preprocessing, no possible gap between what we
think we measure and what we measure. The modest size of the model becomes an
asset: experiments take minutes, not days, which makes it possible to measure a
whole curve rather than a single point.

### 1.2 The thread

The work follows a simple arc, and each step is a failure of the previous one.

1. **Build**: a hand-made CNN, 98.6% on MNIST.
2. **Attack**: FGSM drops it to 1.8%, PGD to 0.0%. Clean accuracy says nothing
   about robustness.
3. **Defend**: adversarial training raises robustness, but reveals an unexpected
   structure — a bell-shaped curve whose optimum is *inside* the range, not at
   the highest attack strength.
4. **Break your own defence**: while looking for the mechanism, we find that
   what matters is not the strength of the training attack but **the order in
   which the difficulty is presented**.

### 1.3 What this work contributes

- **A simple, verified, counter-intuitive law**: with identical inner budgets
  and identical cost, the scheduling of the inner-attack budget makes a
  22-point difference on MNIST and a 41-point difference on KMNIST.
- **Its replication** on a second dataset, with what transfers (the law) and
  what does not (the level, -23.9 points).
- **An honest measurement method**: AutoAttack as the judge, our suite as a
  diagnostic, and our own bias quantified rather than hidden.
- **An extension**: making the budget adaptive per batch adds 7 to 9 official
  points for a few extra minutes.

What this work does not claim to be: a CIFAR-10 state of the art, a new
theoretical bound, or a robustness record. It is a work of **mechanism**, on a
small model, with reproducible numbers and an explicit anchor in the literature
(S.5.1).

---

## 2. The playing field and how we measure

### 2.1 Two engines, the same mathematics

The project contains two implementations of the same network: the hand-made
NumPy version, and a PyTorch port used to train faster (about nine times faster
on CPU, more on GPU). Weights are interchangeable in `.npz` format, and the
architecture is strictly identical everywhere: two convolutional layers (32 then
64 channels), two max-poolings, one dense layer, 421,642 parameters. No
comparison in this document mixes two architectures.

The threat model is the same from beginning to end: L-infinity perturbations of
at most **eps = 0.30**, on images normalized to `[0,1]`. This is not an arbitrary
choice: **it is the MNIST testbed value used in the literature**, by Madry et al.
(2017) and by TRADES (Zhang et al., 2019) — which is exactly what makes the
anchor in S.5.1 legitimate. It is also the regime where an undefended model
collapses to **0.0%**, which makes it an unambiguous ground for observing defence
mechanisms. Whether the law depends on eps (0.1, 0.2) has **not** been tested:
see the limitations. Every evaluation attack is replayed at eps = 0.30, including
for models whose *training* attack used another budget — this is what makes the
rows of the final table comparable with each other.

### 2.2 The six measurement rules

Half of the work consisted in learning how to measure. Each of the six rules
below was born from a concrete mistake:

1. **Announce on 10,000 images.** On the same 500 test images used during
   development, every model was pulled upwards (for example 95.6% -> 93.8% for
   one of the adaptive-budget models).
2. **Select on validation, announce on test.** A bug once produced a "better"
   model that was in fact the model from epoch 1.
3. **The judge is AutoAttack.** Our in-house suite is fast and serves to
   diagnose and explain; it does not serve to announce.
4. **Never report a robust radius measured with PGD alone.** On a rough model,
   the PGD radius ranks **backwards**: `abl_b` has the largest PGD radius in the
   series and the fourth worst case. Always measure the radius with PGD *and*
   with Square.
5. **Sensitivity ranks, it does not prove.** A two-second measurement (sensitivity
   of the loss under perturbation inside the ball) ranks models in the same order
   as the official worst case, 5 times out of 5. But a model that masks its
   gradient is flat by construction: this measurement serves to rank and explain,
   never to conclude.
6. **One variable at a time, and the adoption criterion written before the run.**
   The criteria written before the runs are kept in this document, including the
   ones that turned out to be wrong.

### 2.3 The price of honesty, quantified

Our in-house suite (500 images, Square pushed to 3,000 steps) is **optimistic by
2 to 15.5 points** depending on the model, compared with AutoAttack on 10,000
images:

| model | in-house worst case | official worst case | gap |
|---|---|---|---|
| `A6` (plan `0.2 -> 2 eps`) | 85.8% | 82.40% | -3.4 |
| KMNIST `0.2 -> 2 eps` | 62.6% | 58.53% | -4.1 |
| `A8` (plan `0.2 -> 1 eps`) | 84.4% | 78.25% | -6.2 |
| KMNIST `0.2 -> 1 eps` | 58.4% | 51.03% | -7.4 |
| `abl_a` (constant budget, 2 eps) | 63.2% | 50.71% | -12.5 |
| `A9` (plan `2 -> 0.2 eps`) | 63.8% | 50.35% | -13.5 |
| KMNIST `1 -> 0.2 eps` (inverse plan) | 17.0% | 1.49% | **-15.5** |

And this bias is not scattered at random: it **separates the two families of
recipes**. The five models whose recipe starts low are overestimated by 2.4 to
7.4 points; the three recipes that **start high** (`abl_a` -12.5, `A9` -13.5,
KMNIST inverse -15.5) are overestimated by 12.5 to 15.5 points. The two intervals
do not overlap.

**What this means.** Our attack suite is not biased "on average": it is
optimistic **precisely on the family that the law declares bad**. This is an
argument in favour of the law, not against it: the high-start models are exactly
those whose apparent robustness does not survive a stronger attack. It is also
the technical reason behind measurement rule 3: the only figure that can be
announced is the judge's.

---

*Remaining sections (3 to 10 and the appendices) are being translated; the French
file remains the reference until then.*
