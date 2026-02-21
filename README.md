

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

| Layer        | Access Level        | What We Measure                |
| ------------ | ------------------- | ------------------------------ |
| 🔍 White-Box | Internal logits     | Model probability distribution |
| 🟡 Gray-Box  | Token probabilities | Sequence confidence            |
| ⚫ Black-Box  | Only outputs        | Behavioral stability           |
| 🧠 Semantic  | Embeddings          | Meaning-level consistency      |

---

# 🏗️ 2. Why Multi-Level Uncertainty?

Hallucination is fundamentally an uncertainty miscalibration problem.

A model hallucinates when:

$$
Confidence_{model} \gg Correctness
$$

We measure:

1. Distribution sharpness
2. Sequence likelihood
3. Output agreement
4. Semantic stability
5. Calibration error

---

# 🔍 3. White-Box Uncertainty

## 3.1 Predictive Entropy

$$
H(p) = - \sum_{i=1}^{V} p_i \log p_i
$$

Low entropy → high certainty
High entropy → uncertainty

---

## 3.2 Sequence Log Probability

Autoregressive probability:

$$
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{t-1})
$$

Log form:

$$
\log P(y) = \sum_{t=1}^{T} \log P(y_t \mid y_{t-1})
$$

---

# 🟡 4. Gray-Box Uncertainty

## 4.1 Mean Log Probability

$$
\bar{\ell} = \frac{1}{N} \sum_{i=1}^{N} \log p_i
$$

Converted:

$$
P = e^{\bar{\ell}}
$$

---

## 4.2 Self-Consistency Weighting

$$
Confidence = SelfConsistency \cdot P
$$

---

# ⚫ 5. Black-Box Uncertainty

## 5.1 Self-Consistency

$$
Consistency = \frac{C_{max}}{N}
$$

Where:

* $C_{max}$ = count of most frequent response
* $N$ = total responses

---

## 5.2 Response Entropy

$$
H = - \sum_{i=1}^{K} p_i \log p_i
$$

Where:

* $K$ = number of unique responses
* $p_i$ = frequency of response $i$

---

# 🧠 6. Semantic Consistency

Cosine similarity:

$$
sim(a,b) = \frac{a \cdot b}{|a| |b|}
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
Score = w_w White + w_g Gray + w_b Black
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

## 8.2 Expected Calibration Error

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{N}
\left| acc_m - conf_m \right|
$$

High ECE → overconfidence.

---

# 📈 9. Statistical Validation

## Pearson Correlation

$$
r = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$

---

## AUROC

$$
AUROC = P(score_{pos} > score_{neg})
$$

---

# 🧩 10. Conceptual Insight

$$
Hallucination = Overconfidence + Instability + Semantic Drift
$$



