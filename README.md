

---

# 🧠 Hallucination-in-LLM

### Unified Multi-Level Uncertainty Framework for Hallucination Detection

---

# 📌 1. Problem Definition

Large Language Models (LLMs) generate fluent and coherent text —
however, fluency ≠ factual correctness.

This leads to:

> **Hallucination** — confident but factually incorrect outputs.

The key research question of this project:

> ❓ *How can we systematically measure uncertainty across different visibility levels of an LLM?*

Instead of relying on a single signal, this project proposes:

### 🧩 Multi-Perspective Uncertainty Modeling

We measure hallucination risk from **four epistemic layers**:

| Layer        | Access Level        | What We Measure                           |
| ------------ | ------------------- | ----------------------------------------- |
| 🔍 White-Box | Internal logits     | Model’s internal probability distribution |
| 🟡 Gray-Box  | Token probabilities | Confidence of generated sequence          |
| ⚫ Black-Box  | Only outputs        | Behavioral stability                      |
| 🧠 Semantic  | Embeddings          | Meaning-level consistency                 |

This multi-layer structure allows **robust hallucination detection under different deployment scenarios**.

---

# 🏗️ 2. Why Multi-Level Uncertainty?

Hallucination is fundamentally an **uncertainty miscalibration problem**.

A model hallucinates when:

[
Confidence_{model} \gg Correctness
]

Therefore, we need to measure:

1. Internal distribution sharpness
2. Sequence-level likelihood
3. Output-level agreement
4. Semantic stability
5. Confidence calibration

Each mathematical expression in this project corresponds to one of these failure modes.

---

# 🔍 3. White-Box Uncertainty (Internal Epistemic Uncertainty)

White-box assumes we have access to logits.

## 3.1 Predictive Entropy

H(p) = - Σ (p_i * log(p_i))      for i = 1 ... V


### 🔎 Why use entropy?

Entropy measures **distribution spread**.

* Low entropy → model strongly prefers one token → high certainty
* High entropy → probability mass is distributed → uncertainty

If the model is unsure internally, entropy increases.

👉 We use entropy to capture **token-level epistemic uncertainty**.

This is critical because hallucination often happens when:

* The model picks a high-probability token from a flat distribution.

---

## 3.2 Sequence Log Probability

[
\log P(y) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t})
]

### 🔎 Why log-probability?

Because language modeling is autoregressive:

[
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
]

Taking log:

[
\log P(y) = \sum \log P(y_t)
]

We use this to measure:

> How probable does the model think this entire answer is?

Low sequence probability = low internal confidence.

This gives a **global confidence estimate**, not just token-level.

---

# 🟡 4. Gray-Box Uncertainty (Partial Observability)

Gray-box assumes:

* We don’t have logits.
* We only have token probabilities.

## 4.1 Mean Log Probability

[
\bar{\ell} = \frac{1}{N} \sum_{i=1}^{N} \log p_i
]

### 🔎 Why average?

Longer sequences naturally have lower joint probability.

Averaging removes length bias.

We then exponentiate:

[
P = e^{\bar{\ell}}
]

This gives a normalized confidence measure.

---

## 4.2 Self-Consistency Weighting

[
Confidence = SelfConsistency \times P
]

Why multiply?

Because probability alone is insufficient.

If multiple generations disagree,
confidence should decrease.

This bridges gray-box and behavioral stability.

---

# ⚫ 5. Black-Box Uncertainty (Deployment-Level Safety)

Assume:

* No logits
* No probabilities
* Only final responses

This simulates API-only environments (e.g., production LLMs).

---

## 5.1 Self-Consistency

[
Consistency = \frac{\text{Most Common Response}}{\text{Total Responses}}
]

### 🔎 Why?

If the model answers differently each time,
it signals instability.

Hallucination often correlates with low agreement across samples.

---

## 5.2 Response Entropy

[
H = - \sum_{i=1}^{K} p_i \log p_i
]

Measures distribution of responses.

High entropy → disagreement
Low entropy → stable behavior

This captures **behavioral epistemic uncertainty**.

---

# 🧠 6. Semantic Consistency (Meaning-Level Robustness)

Surface text may differ but meaning can remain stable.

We compute cosine similarity:

[
sim(a,b) = \frac{a \cdot b}{|a||b|}
]

Then average:

[
S = \frac{1}{N} \sum_{i<j} sim(e_i, e_j)
]

Uncertainty:

[
U = 1 - S
]

### 🔎 Why embeddings?

Because hallucination may not be visible in lexical form.

Semantic instability reveals deeper inconsistency.

This captures **representation-level uncertainty**.

---

# 🎯 7. Final Hallucination Score

[
Score = w_w \cdot White + w_g \cdot Gray + w_b \cdot Black
]

### Why weighted combination?

Because:

* White-box captures internal epistemics.
* Gray-box captures token-level confidence.
* Black-box captures behavioral stability.

Each sees a different failure mode.

Default weights:

| Module | Weight | Rationale                    |
| ------ | ------ | ---------------------------- |
| White  | 0.4    | Most direct epistemic signal |
| Gray   | 0.3    | Sequence confidence          |
| Black  | 0.3    | Deployment realism           |

This creates a **risk-aware unified score**.

---

# 📊 8. Calibration Theory

Hallucination is not only about uncertainty —
it is about **miscalibration**.

## 8.1 Brier Score

[
BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
]

Measures:

> Are predicted confidences numerically aligned with reality?

Lower = better calibrated.

---

## 8.2 Expected Calibration Error (ECE)

[
ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} | acc(B_m) - conf(B_m) |
]

Measures gap between:

* Confidence
* True accuracy

If ECE is high → model is overconfident.

Hallucination = overconfidence failure.

---

# 📈 9. Statistical Validation

## Pearson Correlation

[
r = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
]

Checks:

> Does uncertainty correlate with actual errors?

---

## AUROC

Probability that a hallucinated sample gets higher uncertainty score than a correct sample.

Measures ranking quality.

---

# 🎓 10. Research Contributions

This project contributes:

* Multi-layer epistemic modeling
* Calibration-aware hallucination scoring
* Deployment-adaptive uncertainty modules
* Statistical reliability validation
* Modular research architecture

This is suitable for:

* 📄 Academic paper
* 🎓 Master's thesis
* 🇹🇷 TÜBİTAK 2209 / 1001 project
* 🤖 Production AI risk system

---

# 🧩 11. Conceptual Insight

Hallucination is not randomness.

It is:

[
Overconfidence + Instability + Semantic Drift
]

This framework measures all three.

---

