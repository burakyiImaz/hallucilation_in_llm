
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

$$
Confidence_{model} \gg Correctness
$$

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

$$
H(p) = - \sum_{i=1}^{V} p_i \log p_i
$$

### 🔎 Why use entropy?

Entropy measures **distribution spread**.

* Low entropy → model strongly prefers one token → high certainty
* High entropy → probability mass is distributed → uncertainty

If the model is unsure internally, entropy increases.

👉 We use entropy to capture **token-level epistemic uncertainty**.

---

## 3.2 Sequence Log Probability

Autoregressive probability:

$$
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{<t})
$$

Log form:

$$
\log P(y) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t})
$$

We use this to measure:

> How probable does the model think this entire answer is?

Low sequence probability = low internal confidence.

---

# 🟡 4. Gray-Box Uncertainty (Partial Observability)

Gray-box assumes:

* We don’t have logits.
* We only have token probabilities.

## 4.1 Mean Log Probability

$$
\bar{\ell} = \frac{1}{N} \sum_{i=1}^{N} \log p_i
$$

Converted back to probability:

$$
P = e^{\bar{\ell}}
$$

---

## 4.2 Self-Consistency Weighting

$$
Confidence = SelfConsistency \times P
$$

---

# ⚫ 5. Black-Box Uncertainty (Deployment-Level Safety)

Assume:

* No logits
* No probabilities
* Only final responses

## 5.1 Self-Consistency

$$
Consistency = \frac{\text{Most Common Response}}{\text{Total Responses}}
$$

---

## 5.2 Response Entropy

$$
H = - \sum_{i=1}^{K} p_i \log p_i
$$

Where:

* $K$ = number of unique responses
* $p_i$ = frequency of response $i$

---

# 🧠 6. Semantic Consistency (Meaning-Level Robustness)

Cosine similarity:

$$
sim(a,b) = \frac{a \cdot b}{|a||b|}
$$

Semantic consistency score:

$$
S = \frac{1}{N} \sum_{i<j} sim(e_i, e_j)
$$

Uncertainty:

$$
U = 1 - S
$$

---

# 🎯 7. Final Hallucination Score

$$
Score = w_w \cdot White + w_g \cdot Gray + w_b \cdot Black
$$

Default weights:

$$
w_w = 0.4, \quad w_g = 0.3, \quad w_b = 0.3
$$

---

# 📊 8. Calibration Theory

## 8.1 Brier Score

$$
BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

---

## 8.2 Expected Calibration Error (ECE)

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| acc(B_m) - conf(B_m) \right|
$$

If ECE is high → model is overconfident.

Hallucination = overconfidence failure.

---

# 📈 9. Statistical Validation

## Pearson Correlation

$$
r = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$

---

## AUROC

$$
AUROC = P(score_{positive} > score_{negative})
$$

---

# 🧩 11. Conceptual Insight

$$
Hallucination = Overconfidence + Instability + Semantic Drift
$$

---

