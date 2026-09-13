# The order in which you present the difficulty is what builds robustness

**A law about the displacement budget of the inner attack: found, replicated on a
second dataset, and passed to the official judge.**

*Complete English version of `ARTICLE-fr.md` (2026-09-13); the French file is the
reference if the two ever diverge. Figures live in `figures/` and are
language-neutral (English labels). Sources of every number: `adversarial/memoire.md`
and the JSON files in `adversarial/results/logs/`.*

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

## 3. Attacking: breaking the clean model

The hand-made model reaches 98.6% on MNIST. A single gradient step (FGSM,
eps = 0.30) is enough to bring it down to **1.8%**. Twenty PGD steps bring it
down to **0.0%**.

The attack suite of the project covers several families: gradient-based (FGSM,
PGD, APGD-CE, APGD-DLR), minimum-distance (CW-L2), decision-based (Boundary),
gradient-free (Square, NES), escaping obfuscated gradients (BPDA + EOT), and a
certified defence (randomized smoothing, S.8.1).

**The first interesting result is not the drop, it is the agreement between the
families.** On a healthy model, the attacks converge to within a few points. The
best model on MNIST (`0.2 -> 1 eps` plan) gives, on the same 500 images:
FGSM 95.2%, PGD-20 89.2%, APGD-CE 84.4%, APGD-DLR 84.4%, Square 84.6%. Four
families, one of them gradient-free, within less than one point of each other.

On a model affected by "masking", they contradict each other. The model `abl_a`
gives 85.4% under APGD-DLR and **63.2%** under Square: a 22-point gap between a
gradient attack and a gradient-free attack. This disagreement is a **symptom**,
not noise: it served as a compass throughout the whole project, before having
access to the official judge.

Finally, "naive" adversarial training (hardening against FGSM alone) gives the
illusion of a defence: the model resists FGSM and remains broken by PGD.
Hardening is not a matter of brute force, it is a matter of recipe. That is the
subject of what follows.

---

## 4. Defending: the bell-shaped curve

### 4.1 The first hardening, and the lesson of the epochs

The first hardened version (NumPy, PGD with 7 steps of size eps/4, 60,000
images) rises to 67.2% clean accuracy and falls back to **1.2%** worst case:
barely better than nothing. The next campaign (three recipes, warm start, 60,000
images) brings a useful lesson: the run judged bad after 10 epochs reaches
**91.0%** after 120 epochs. Before declaring a recipe bad, it must be given its
budget of epochs. This document now applies 120 epochs everywhere.

### 4.2 The step ablation: the peak is inside

Everything is kept fixed (120 epochs, PGD-20, eps = 0.30, augmentation, 60,000
images, seed 42) and only the **fineness of the step** of the inner attack is
changed. The displacement budget, that is to say the maximal distance the attack
can travel, is `step x number of steps`.

| recipe | inner budget | worst case (in-house) | clean accuracy |
|---|---|---|---|
| `run B` | 1.25 eps (step eps/4) | 42.0% | 99.6% |
| `abl_b` | 5 eps (step eps/4) | 38.8% | 99.8% |
| `A4` | 0.5 eps (step eps/10) | 49.2% | 99.4% |
| `v4` / `abl_a` | 2 eps (step eps/10) | 61.8% / **63.2%** | 99.4% / 99.6% |
| `abl_c` | 20 eps (step eps) | **1.6%** | 98.4% |

The learned robustness is maximal for an **intermediate** budget: neither too
soft, nor too hard. Too soft, the training learns a masked robustness (`run B`
shows 91% under PGD-20 and 42% worst case). Too hard, the training locks up.

`abl_c` is the perfect counter-example, and the most instructive one: **98.4%
clean accuracy**, a model that looks intact, for **1.6%** worst case. The log
explains why, and the figure is unambiguous: the fooling rate of the inner
attack collapses from 90.8% at epoch 1 to **2.2%** at the end of the run, and
the adversarial cross-entropy joins the clean cross-entropy (0.068 against
0.022). In other words, the "adversarial" examples are no longer adversarial at
all: with a step the size of eps, the attack jumps over the ball and lands on
points that the model classifies correctly. The run **degenerates into clean
training**, and the model ends up with no robustness at all (val PGD10: 0.0%).

FIGURE 1 (`figures/fig1_budget_bell.png`, labels in English): the worst case as
a function of the budget of the inner attack. Blue line: constant budgets (the
peak is at ~2 eps). Arrows: the two budget plans, green for a low start (the
good one), red for a high start.

### 4.3 The step matters as much as the budget

One detail almost went unnoticed: `A4` (fine step eps/10, 5 steps, budget
0.5 eps) obtains 49.2% where `run B` (coarse step eps/4, budget 1.25 eps)
obtains only 42.0%. The previous ablation confused two axes, the **fineness** of
the step and the **number** of steps. The formulation we keep is: what matters
is the **legibility** of the perturbation, not only its strength. A fine step
accumulates coherent gradient directions; a coarse step produces a perturbation
that the model cannot learn to counter.

### 4.4 The central result: it is the ORDER

**The exact shape of the plans.** A budget plan is a **linear ramp over the
epochs**: at epoch `t`, the budget is `deb + (fin - deb) x (t-1)/(epochs-1)`,
expressed in multiples of eps. The step of the inner attack remains **fixed**
(eps/10): only the **number of steps** is changed, hence
`budget = step x alpha`. Concretely, `--plan-budget "0.2,2"` takes the inner
attack from 2 steps (0.2 eps) at epoch 1 to 20 steps (2 eps) at epoch 120, in a
single monotone ramp. Two plans with the same bounds but another ramp shape
(logarithmic, step-wise, power law) have **not** been tested: the linear shape
was fixed, not compared (see the limitations).

The most interesting question remains: at a given budget, does the order in
which the budgets are presented to the model matter? The decisive experiment
consists of two runs, of identical cost, with exactly the same set of budgets
traversed:

| recipe | budgets traversed | cost | worst case (in-house) |
|---|---|---|---|
| `A6`: plan `0.2 -> 2 eps` | 2 .. 20 steps | 11 min | **85.8%** |
| `A9`: plan `2 -> 0.2 eps` | 20 .. 2 steps | 11 min | **63.8%** |

**Same amount of compute, same budgets, reversed order: a 22-point gap.** The
increasing plan reaches its maximal budget (the hardest one) *later*; the
decreasing plan reaches it straight away. And this observation is not isolated:
all the bad models of the series (`abl_a` 63.2%, `A7` 62.0%, `A9` 63.8%) have a
**high start**, all the good ones (`A6` 85.8%, `A8` 84.4%, `A1` 93.6%) start at
2 steps (0.2 eps).

The proposed reading: a start at a short budget plays the role of a
regularization that keeps the inner attack **informative** — it still fools the
model, so it provides a useful signal — whereas starting at high strength
places the optimization straight away in the locking regime observed on `abl_c`.
A neighbouring theoretical lead is the **min-min** objective of FAT (Wong et al.,
ICML 2020): the early stopping of the inner attack changes the nature of the
optimization problem.

**Two caveats on this result, to be written in black and white.**

1. **The flagship figure, now official on both sides.** `A6` and `A9` have both
   gone through the official judge:

| set | increasing plan | inverse plan | in-house gap | **official gap** |
|---|---|---|---|---|
| MNIST | `0.2 -> 2 eps`: **82.40%** | `2 -> 0.2 eps`: **50.35%** | 22.0 | **32.1** |
| KMNIST | `0.2 -> 1 eps`: **51.03%** | `1 -> 0.2 eps`: **1.49%** | 41.0 | **49.5** |

Both gaps **grow** under the unbiased judge. The natural objection ("what if the
22-point gap were only an artefact of your optimistic suite?") therefore turns
around: it is the opposite, and by 10 points on MNIST as well as 8.5 points on
KMNIST. A prediction had been written before the measurement of `A9` (48 to
58%): 50.35% falls inside it.
2. **The proposed mechanism remains a reading, but it is now measured.**
S.4.5 exploits the trajectories recorded in the logs (no additional run) and
replaces "the low start preserves the information" with a quantified criterion:
intermediate fooling and unsaturated adversarial cross-entropy. What is still
not done is the **intervention** that would isolate the mechanism from an
obvious alternative — a budget schedule that interacts with the learning rate
program, independently of the informative content of the gradient. The test that
would settle it is a **non-monotone** plan (`0.2 -> 2 -> 0.2 eps` against
`2 -> 0.2 -> 2 eps`): it distinguishes "monotone growth" from "soft start". It is
cheap and left to later work.

**A caveat on the 22 points: the figure survives a seed sweep.** Each recipe was
initially only a single run, which left the obvious objection open: what if the
gap came from the initialization or from the order of the mini-batches? The pair
was therefore re-run with **three additional seeds** (1, 2, 3), seed 42 already
existing on both sides:

| recipe (same budgets, same cost) | seeds | in-house (min-max) | mean | std dev |
|---|---|---|---|---|
| plan `0.2 -> 2 eps` (increasing) | 42, 1, 2, 3 | 85.6 to 88.6 | **86.9** | 1.4 |
| plan `2 -> 0.2 eps` (decreasing) | 42, 1, 2, 3 | 57.2 to 63.8 | **61.2** | 3.1 |

- gap of the **means**: **25.7 points**;
- **minimal** gap observed, all seeds together (the worst increasing seed
  against the best decreasing seed): **21.8 points** — so never less than the
  22 points announced;
- maximal gap: 31.4 points;
- the minimal gap is worth **7 times** the largest of the two standard
  deviations.

In other words, the dispersion between seeds (1.4 and 3.1 points) is an order of
magnitude below the effect. A notable fact: it is the **decreasing** recipe that
is the most unstable (standard deviation 3.1 against 1.4), which is consistent
with the idea that a high start places the training near a cliff.

**A caveat on the 22 points: two independent measurements, not one.** The effect
was reproduced on a second dataset with distinct runs (S.7.3), where the gap
reaches 41 points. It is not a repetition of seeds on the same configuration,
but a repetition of the **effect** on other data, which answers the most common
objection.

The same experiment, with cheaper plans, served as a guard rail: a
`0.2 -> 0.5 eps` plan (6 min) gives only 31.8% and a `0.05 -> 0.2 eps` plan
(5 min) 0.0% — lowering the *ceiling* destroys the robustness. It is the
**start** that must be low, not the ceiling.

FIGURE 2 (`figures/fig2_order.png`): the two budget plans, same budgets, same
cost, reversed order — in-house (500 images) and at the official judge
(AutoAttack, 10,000 images). Both gaps, and the fact that the OFFICIAL gap is
larger, are annotated on the figure.

### 4.5 What the logs tell us: two ways to fail, one way to succeed

The training logs (kept in the repository, `adversarial/results/logs/`) contain,
at every epoch, two numbers that are enough to diagnose a run: the **fooling
rate** of the inner attack (which share of the adversarial batch it flipped) and
the **adversarial cross-entropy** measured on that batch.

| recipe (KMNIST) | fooling start -> end | final adv CE | final val PGD10 | worst case |
|---|---|---|---|---|
| plan `0.2 -> 2 eps` | 35.2% -> **47.3%** | 1.38 | 89.2% | **62.6%** |
| plan `0.2 -> 1 eps` | 35.2% -> **40.9%** | 1.18 | 85.2% | **58.4%** |
| plan `0.2 -> 0.5 eps` | 35.2% -> 27.6% | 0.79 | 67.2% | 31.8% |
| inverse plan `1 -> 0.2 eps` | 96.5% -> 16.4% | 0.50 | 47.6% | 17.0% |
| constant 1 eps | 96.5% -> 44.1% | 1.27 | 79.1% | 15.2% |
| constant 2 eps | 97.2% -> 90.2% | **2.322** | 9.0% | 1.2% |
| plan `0.05 -> 0.2 eps` | 17.3% -> 14.2% | 0.41 | 3.7% | 0.0% |

Three readings.

**The training signal is set by the left-hand budget.** All these runs start
from the same weights (identical warm start). At epoch 1, the recipes that start
at 2 steps (0.2 eps) see an adversarial cross-entropy of **1.234** and a fooling
rate of 35%; those that start at 10 steps (1 eps) see **3.895** and 96.5%. The
difference is therefore not "less compute": it is the **nature of the
examples**. An adversarial loss of 3.9 (above `ln 10 = 2.303`) means that the
adversarial examples are incomprehensible to the model at the moment they are
presented.

**Two ways to fail a run, and they are visible without evaluating the model.**

1. *The inner attack switches off* (`abl_c`, step the size of eps): the fooling
   rate falls to **2.2%** and the adversarial cross-entropy joins the clean one
   (0.068 against 0.022). The "adversarial" examples are no longer adversarial;
   the run degenerates into clean training; final robustness 0%.
2. *The model saturates to the uniform* (constant 2 eps on KMNIST, and `v5`
   whose inner attack is an APGD): the attack keeps fooling 88 to 90% of the
   batch, but the adversarial cross-entropy **blocks exactly at `ln 10 = 2.32`**
   — the model output is uniform at the adversarial point, so the gradient no
   longer tells it anything. Corresponding plateaus: val PGD10 9.0% and 11.6%.

**One way to succeed: keep the attack informative without saturating the
model.** The good recipes maintain an **intermediate** fooling rate (35 to 47%)
and a **moderate** adversarial cross-entropy (1.2 to 1.4), and their fooling
rate **rises** when the budget rises (35 -> 47% for `0.2 -> 2`): the model is
shaken, and it keeps learning. Conversely, all the recipes with a high start see
their fooling rate **fall** (96.5 -> 16.4; 96.5 -> 44.1) — the fixed attack
ceases to be a challenge, and the model obtains a robustness that holds only
against that precise attack.

**What this changes in the article.** The mechanism is no longer a vague reading
("the low start preserves the information") but a measurable, free criterion,
read from the logs at every epoch: if the fooling rate collapses towards 2%, the
run is lost; if the adversarial cross-entropy blocks at 2.303, it is lost too;
if it stays in the intermediate zone, it has a chance. It is still not a proof
(a masked model can show a healthy profile: `abl_a` ends at 41% fooling, 1.21
adversarial loss and 91.3% val PGD10 for an official worst case of 50.71%), but
it is a filter that eliminates in one log line the two degeneracies that make a
run useless.

FIGURE 5 (`figures/fig5_trajectoires.png`): the two panels, read directly from
the logs of the repository.

### 4.6 The extension: an adaptive budget, batch by batch

The deterministic plan is a curve fixed in advance. A variant consists in
aiming, **for each batch**, at a target difficulty and stopping the inner attack
as soon as it is reached, at step `k*`. The overhead is zero: at fixed step, the
PGD trajectory already contains all the candidates.

**Exact parameterization**, so that the result is reproducible and open to
criticism: the target is a **fooling rate** (fraction of the batch that the
inner attack must have flipped), fixed at **0.5**; the attack stops at the first
step `k*` such that the fooling rate reaches 0.5, and the remaining steps are
saved. The target itself follows a ramp (0.2 at the beginning, 0.5 from the
middle of the run), because a still weak model cannot fool half a batch at epoch
1. A guard rail warns if the target is reached only at the step ceiling on too
large a fraction of the batches (a sign that we are outside the feasible window).

Result: **93.6% in-house / 91.25% official**, against 82.40% for the best
deterministic plan. The batch-wise servo control therefore brings +7 to +9
official points for about ten extra minutes.

**What has not been tested on this point**: the value of the target. A
smoother-target variant (relative cross-entropy) and a control at a very high
target (0.9, which should degrade) are described in the documentation of the
attack but were not launched: the announced result therefore holds only for the
target 0.5, and its sensitivity to this parameter remains an open question. It
is the best model of the project, and it is also the one whose parameterization
is the least ablated: to be fixed before any publication that would put it
forward.

A more robust model must also survive the masking audit, without which the gain
would be an artefact. Four checks, all passed: the outputs are not saturated;
the gradient is informative (the gradient/random ratio is 8.5 times better than
chance); the transfer from another model does not beat the white-box attack; and
the model does collapse at eps = 0.5 under a fine-step PGD (0.6%), which proves
a real robust radius and not an artificially flat surface.

### 4.7 Two acknowledged corrections

Two intuitions were written, tested, then abandoned. They are part of the
result:

1. "The robust radii are equal, so it is a window effect": **false**. The radius
   measured with PGD alone is invalid when the gradient no longer guides the
   attack (rule 4).
2. "The gradient/random ratio predicts the size of the gap between attacks":
   **false** as well. The counter-example `abl_c` has a ratio of 1.16 and a
   *negative* gap. This ratio only predicts whether the gradient attacks are
   usable.

---

## 5. Placing the result, and what remains open

### 5.1 Where we stand relative to the literature

On MNIST at eps = 0.30, the reference implementations report:

| model | clean | robust |
|---|---|---|
| Madry et al. 2017, PGD-40 | 99.36% | **96.01%** |
| TRADES, Zhang et al. 2019 (`1/lambda = 6`) | 99.48% | **95.60%** |
| **our** TRADES (non-conforming variant, see S.5.2) | 93.26% | 25.18% |
| our best (`bande_cible50`), 421k parameters, 120 epochs | 98.85% | **91.25%** |
| our `A6` (plan `0.2 -> 2 eps`) | 99.17% | **82.40%** |
| our `abl_a` (**same** architecture, constant budget 2 eps) | 99.53% | **50.71%** |

Our TRADES rows appear here for transparency, but they must **not** be read as a
method comparison point: it is established in S.5.2 that our implementation does
not reproduce the reference, and the gap (~70 points) is far too large to be a
method effect.

Two sentences suffice. First, our best model is about **five points** from the
reference, with a much smaller architecture and 120 epochs: the figures of this
document are not out of touch with reality. Second, and that is the subject:
**within the same architecture**, the choice of recipe makes a 45-point
difference (50.71% against 91.25%). The article does not claim a record; it
explains that gap.

### 5.2 What remained open

**Our implementation of TRADES does not reproduce the reference.** Two runs,
`beta 2` with warm start and `beta 6` from scratch, give **22.01%** and
**25.18%** official robustness, with clean accuracies of 96.5% and 93.3% — that
is to say **lower** than those of our models trained with cross-entropy (99.2%
to 99.6%), whereas the CE term of TRADES is precisely supposed to protect the
clean accuracy. Two deviations from the reference are identified in the code:
the **direction of the KL** (we measure `KL(p_adv || p_clean)` where the
reference minimizes `KL(p_clean || p_adv)`) and the fact that **the inner attack
maximizes the cross-entropy** instead of the KL term.

The gap of ~70 points with the reported 95.60% is too large to be read as a
comparison of methods: these two runs are published as an **acknowledged
limit**, not as a result on the loss. Any sentence of the type "the order
compensates for a badly tuned loss" would be unsupported by these measurements.

---

## 6. The official judge

All the figures announced in this document come from AutoAttack (`standard`
version: APGD-CE, APGD-T, FAB-T, Square), on 10,000 images, at eps = 0.30. Eight
models were put through the judge (the two matched-recipe pairs are among them,
which is what makes the flagship result verifiable):

| model | recipe | in-house | official | gap |
|---|---|---|---|---|
| `bande_cible50` | adaptive budget | 93.6% | **91.25%** | -2.4 |
| `A6` | plan `0.2 -> 2 eps` | 85.8% | **82.40%** | -3.4 |
| `A8` | plan `0.2 -> 1 eps` | 84.4% | **78.25%** | -6.2 |
| KMNIST `0.2 -> 2 eps` | plan `0.2 -> 2 eps` | 62.6% | **58.53%** | -4.1 |
| KMNIST `0.2 -> 1 eps` | plan `0.2 -> 1 eps` | 58.4% | **51.03%** | -7.4 |
| `abl_a` | constant budget 2 eps | 63.2% | **50.71%** | -12.5 |
| `A9` | plan `2 -> 0.2 eps` | 63.8% | **50.35%** | -13.5 |
| KMNIST `1 -> 0.2 eps` | inverse plan | 17.0% | **1.49%** | **-15.5** |

Three observations.

**The bias of our suite is always in the same direction, and it separates the
two families of recipes.** The five models with a low start are overestimated by
2.4 to 7.4 points; the three recipes with a **high start** (`abl_a`: -12.5,
`A9`: -13.5, KMNIST inverse plan: -15.5) are overestimated by 12.5 to 15.5
points. The two intervals do not overlap. Our suite is therefore not biased "on
average": it is optimistic **precisely on the family that the law designates as
bad**. This is an argument in favour of the law: the recipes with a high start
are exactly those whose apparent robustness does not survive a stronger attack.

**The ranking is preserved on the healthy models, and it degrades exactly on the
family that the article declares bad.** `abl_a` goes from fifth in-house to
**sixth** officially, `A9` from fourth to **seventh**, and the KMNIST inverse
plan falls from 17.0% to **1.49%**. Conversely, the recipes with a low start
(`A1`, `A6`, `A8`, and the two KMNIST) keep exactly the same order. Formulation
to keep: *the in-house suite ranks the healthy models well and errs in favour of
the suspects*.

**The official figures themselves are of different quality.** AutoAttack issued
a warning on the KMNIST `0.2 -> 1 eps` model ("Square Attack reduced the robust
accuracy by 2.24%"), which means that the 51.03% is itself slightly optimistic.
No warning on the 58.53%. The two KMNIST figures are therefore not of identical
quality, and the model card will say so.

FIGURE 3 (`figures/fig3_inhouse_vs_official.png`): in-house worst case (x)
against official worst case (y), each point numbered and listed in the legend
(`abl_a` and `A9` are merged: they fall on the same point). Green: recipes with
a low start, red: high start. The green points are near the diagonal, the red
ones detach from it downwards.

---

## 7. Replication: KMNIST, the Japan anchor

### 7.1 Why a second set

KMNIST (Kuzushiji-MNIST) contains Japanese cursive characters, in a format
identical to MNIST: 28x28 images, ten classes, 60,000 training images, 10,000
test images. Same dimensions, same pipeline, same model, same eps: it is a
**free** second set for testing the generality of the law. It is also a coherent
hook for a work carried out from Japan.

Before launching anything, the prediction was written: "hard ~60-65%, soft
~84%, inverse order -20 points". It turned out half wrong, and that is
instructive: the predictions of *level* were wrong, the prediction of *order*
was right.

### 7.2 Eight trained models, nine measurement points

| recipe | inner budget | worst case (in-house) | official |
|---|---|---|---|
| clean model (no attack) | - | 0.0% | - |
| constant 2 eps | 20 steps | 1.2% | - |
| inverse plan `1 -> 0.2 eps` | 20 .. 2 steps | 17.0% | **1.49%** |
| constant 1 eps | 10 steps | 15.2% | - |
| plan `0.05 -> 0.2 eps` | 1 .. 2 steps | 0.0% | - |
| plan `0.1 -> 0.5 eps` | 1 .. 5 steps | 25.2% | - |
| plan `0.2 -> 0.5 eps` | 2 .. 5 steps | 31.8% | - |
| plan `0.2 -> 1 eps` | 2 .. 10 steps | 58.4% | **51.03%** |
| plan `0.2 -> 2 eps` | 2 .. 20 steps | **62.6%** | **58.53%** |

### 7.3 What transfers

**The law of the order, and it amplifies.** The increasing plan `0.2 -> 1 eps`
gives 58.4% and its inverse `1 -> 0.2 eps` 17.0%: **a 41-point gap** in-house,
against 22 on MNIST. And above all, this pair is the only one in the project
whose two sides have gone through the official judge: **51.03% against 1.49%,
that is a 49.5-point official gap** — an even larger gap than the one measured
by our suite. Better still: the two models with a low ceiling (`0.1 -> 0.5`:
25.2% and `0.05 -> 0.2`: 0.0%) confirm, on a second set, that it is the *start*
that must be lowered and not the ceiling. The best plan is also the same as on
MNIST.

### 7.4 What does not transfer

**The level.** The two pairs with an identical recipe, measured officially on
both sides:

| recipe | MNIST | KMNIST | gap |
|---|---|---|---|
| plan `0.2 -> 2 eps` | 82.40% | 58.53% | **-23.9 points** |
| plan `0.2 -> 1 eps` | 78.25% | 51.03% | **-27.2 points** |

*The recipe transfers, the level does not.* That is the sentence this work was
trying to be able to write, and it is now supported by two official pairs.

The mechanism of the shift is identifiable: at eps = 0.30, the inner attack is
relatively much stronger on KMNIST. On the clean model, a PGD at 0.1 brings
MNIST down to 44.4% and KMNIST to **9.2%**. As a result, the reference recipe on
MNIST (constant budget 2 eps, 63.2% worst case) **collapses** on KMNIST to
**1.2%** — the budget becomes "out of reach" for the set, and the model makes
the only trade-off left to it.

The log of this run is eloquent: the inner attack fools 90% of the batch from
beginning to end (so it is *not* the locking of `abl_c`), the clean validation
accuracy rises to 99.5% — better than the clean reference model — while the
robustness remains stuck. A clean model *more* accurate and simultaneously
useless: that is exactly the profile of an objective that has become out of
reach.

**The window of feasible budgets, and its consequence for the scope of the
law.** The set of KMNIST measurements empirically delimits a window: a plan whose
**start** is low (2 steps, 0.2 eps) and whose **ceiling** reaches 1 to 2 eps
gives 58 to 63%; a ceiling below 0.5 eps produces only a tiny-radius robustness;
a **constant** budget of 2 eps falls out of reach and collapses to 1.2%. The law
of the order therefore does not apply "in the absolute": **it applies inside a
window of feasible budgets, and this window depends on the dataset** (it is
lower on MNIST, higher on KMNIST, where the constant 2 eps budget fails whereas
it reaches 63.2% on MNIST). This is an important precision for the reader: the
law is not a theorem, it is a conditional regularity, whose condition is
precisely the point to check before transposing a recipe.

**Methodological lesson**: a peak of a curve is not a number, it is a number **
for a given dataset**. Any recipe published without its cross-validation set is
a recipe that has not been tested.

FIGURE 4 (`figures/fig4_recipe_transfers.png`): the two recipes present
identically on the two sets, measured officially on both sides. Same recipe, two
heights: the recipe transfers, the level does not.

---

## 8. What holds by guarantee, and the limits

### 8.1 The only guarantee of the project

All the previous results are **empirical**: we attack, we look at what remains.
Randomized smoothing (Cohen, Rosenfeld and Kolter, 2019) provides a
**guarantee**. We train a base classifier on noisy images `N(0, sigma^2)`, then
build a smoothed classifier `g(x) = argmax_c P(f(x + noise) = c)`. A theorem
then gives a guaranteed L2 radius.

Measured at `sigma = 0.5`, on 1,000 images, with exact confidence bounds
(Clopper-Pearson) and `alpha = 0.001`:

| | value |
|---|---|
| accuracy of the smoothed classifier | 99.0% |
| abstention rate | 0.5% |
| **median certified L2 radius** | **1.214** |
| certified accuracy at R = 0.30 | 98.5% |
| certified accuracy at R = 1.00 | 83.5% |

It is the only figure in this document that is a **guarantee** and not an
observation: no perturbation of L2 norm less than or equal to the certified
radius can change the prediction, by construction. To compare, while staying in
the same norm, with the L2 distances that our CW-L2 attacks must reach in order
to fool the hardened models (from 0.005 to 1.8 depending on the model): the
smoothed model is far ahead.

**Reading trap, never to miss.** The L-infinity ball of radius 0.30 is **not**
included in the L2 ball of radius 1.214: a perturbation of L-infinity norm 0.30
can have an L2 norm up to `0.30 x sqrt(784) = 8.4`. The L-infinity results of
this document and this L2 guarantee therefore answer **two different questions**:
they do not compare and do not replace each other.

### 8.2 Limits

- **The regime tested is narrow**: small images (28x28), two sets only, a single
  eps (0.30), one architecture (421,642 parameters), a single training engine.
  The law is verified on this regime, not demonstrated beyond it. Explicitly out
  of scope: model ensembles, training *against* Square, models beyond 1.7M
  parameters, and transfer between datasets.
- **Our in-house suite is optimistic by 2 to 15.5 points** and errs in favour of
  the suspect models. It serves to rank and to explain, never to announce.
- **The official KMNIST `0.2 -> 1 eps` figure (51.03%) is itself optimistic**,
  AutoAttack signalling a possible improvement of 2.24% under Square. Announced
  limit, to be written in the model card.
- **Empirical robustness, not certified**, except for the smoothed model
  (S.8.1).
- **Our TRADES does not reproduce the reference** (S.5.2): published as a limit,
  the two deviations being identified.
- **A single replication set** and a single winning plan per set: the law of the
  order is reproduced twice (MNIST, KMNIST), not demonstrated.
- **A single eps** (0.30): we do not know whether the law holds at eps = 0.1 or
  0.2, where the window of feasible budgets necessarily shifts (the locking
  becomes harder to trigger, so the high start could become harmless). Cheap
  test (one run per eps), not done.
- **The shape of the ramp has not been compared** (linear against logarithmic
  against step-wise), nor has the value of the target of the adaptive budget
  (S.4.6).
- **Training variance: measured, and it does not threaten the result.** Each
  recipe is trained with several seeds (42, 1, 2, 3): mean 86.9
  (standard deviation 1.4) for the increasing plan, 61.2 (standard deviation
  3.1) for the decreasing plan, and the minimal gap observed between an
  increasing seed and a decreasing seed is **21.8 points**, that is 7 times the
  largest of the two standard deviations (S.4.4). The official figures, for their
  part, were measured only on seed 42 on both sides (82.40% and 50.35%): the
  official dispersion is not measured, but the bias of our suite being
  **systematic** and of known direction (always optimistic, and more so on the
  recipes with a high start), it can only widen the gap, not close it.

---

## 9. Reproducibility

Everything in this document is regenerable. The exact commands of the runs are
in `adversarial/README.md`; the raw results are written as JSON by
`eval_suite.py` and `eval_autoattack.py` into `adversarial/results/logs/`;
`adversarial/torch/tableau_recap.py` rebuilds the final table from these JSON
files, which guarantees that **no figure of this article was copied by hand**.
The logbook (`adversarial/memoire.md`, more than 100 kB) keeps, for each
experiment, the parameters, the result and the criterion that had been written
before launching the run — including the criteria that turned out to be wrong.

The project contains the two engines (S.2.1), the attack suite, the audit scripts
(`audit_masquage.py`, `diag_attaque_interne.py`), the budget plan
(`--plan-budget`), the adaptive-budget attack (`--bande`) and the certification
code. No step depends on an external service.

---

## 10. Conclusion

We built a CNN by hand, broke it in a single gradient step, defended it, then
tried to break our own defence. This path produced a simple result: **it is not
the strength of the training attack that hardens a network, it is the order in
which the difficulty is presented to it.** Twenty-two points of gap on MNIST,
forty-one on KMNIST, for an identical cost.

The second result is a limit, and it matters just as much: **the recipe
transfers, the level does not** (-23.9 points at an identical recipe and with
the official judge). A recipe that works somewhere is not a recipe that works
everywhere.

What follows is editorial before being technical: a model card on Hugging Face
(weights, recipe, robustness per eps, limits), an interactive demonstration
("attack the CNN live"), and an English version of this document.

---

## Appendices

**A. The acknowledged corrections.** Invalid PGD radius in the rough regime;
non-predictive gradient/random ratio; partially wrong KMNIST criterion; wrong
level prediction on KMNIST; the sentence "the order compensates for a badly
tuned loss" withdrawn for lack of experimental support.

**B. Glossary.** FGSM, PGD, APGD, CW-L2, Square, NES, Boundary, BPDA+EOT,
AutoAttack, displacement budget, curriculum, min-min (FAT), gradient masking,
randomized smoothing, certified radius.

**C. State of the document.** Numbers verified and dated; five figures produced
from a single source of numbers (`figures.py`, overlap-checked automatically);
English version written. The detailed plan and the open points are in
`TRAME-fr.md`.
