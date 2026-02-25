
---

#  Hallucination-in-LLM

## A Unified Multi-Level Uncertainty Framework for Hallucination Detection

---

# 1. Introduction: The Core Problem

Large Language Models (LLMs) generate text by predicting the next token based on learned probability distributions. These models often produce outputs that are:

* Grammatically fluent
* Contextually coherent
* Confident in tone

However, fluency and confidence **do not guarantee factual correctness**.

This mismatch gives rise to a phenomenon known as:

> **Hallucination** — confident yet factually incorrect model outputs.

The fundamental research question becomes:

> How can we systematically measure and quantify uncertainty in LLMs to detect hallucinations?

We argue that hallucination is not a single-dimensional problem. Instead, it is a **multi-layer epistemic failure** involving:

* Internal probabilistic uncertainty
* Sequence-level confidence
* Behavioral instability
* Semantic inconsistency
* Calibration mismatch

Therefore, we propose a **Multi-Level Uncertainty Framework**.

---

# 2. Why Multi-Level Uncertainty?

Hallucination is fundamentally a **miscalibrated confidence problem**.

We formalize this idea as:

$$
Confidence_{model} \gg Correctness
$$

In other words, hallucination occurs when the model’s predicted confidence significantly exceeds its actual probability of being correct.

To capture this, we measure uncertainty from four complementary perspectives:

| Layer     | Visibility            | Signal Type                |
| --------- | --------------------- | -------------------------- |
| White-Box | Internal logits       | Distributional uncertainty |
| Gray-Box  | Token probabilities   | Sequence confidence        |
| Black-Box | Output responses only | Behavioral stability       |
| Semantic  | Embeddings            | Meaning-level consistency  |

Each layer captures a different failure mode of the model.

---

# 3. White-Box Uncertainty

White-box access assumes we can inspect the model’s internal logits before sampling.

---

## 3.1 Predictive Entropy

LLMs output a probability distribution over the vocabulary:

$$
p = (p_1, p_2, ..., p_V)
$$

where $V$ is vocabulary size.

We measure uncertainty using **Shannon entropy**:

$$
H(p) = - \sum_{i=1}^{V} p_i \log p_i
$$

### Interpretation

Entropy quantifies the **spread of a probability distribution**.

* If one token has probability ≈ 1 → entropy is low → high certainty
* If all tokens are equally likely → entropy is high → high uncertainty

Entropy measures **distribution sharpness**.

Thus:

* High entropy → internal uncertainty
* Low entropy → internal confidence

---

## 3.2 Sequence Log Probability

A generated sequence

$$
y = (y_1, ..., y_T)
$$

is modeled autoregressively:

$$
P(y) = \prod_{t=1}^{T} P(y_t \mid y_{1:t-1})
$$

Because probabilities are small and products become numerically unstable, we compute in log-space:

$$
\log P(y) = \sum_{t=1}^{T} \log P(y_t \mid y_{1:t-1})
$$

### Why Log?

* Prevents numerical underflow
* Converts multiplication into summation
* Enables stable optimization

Low log-probability implies weak internal support for the generated sequence.

---

# 4. Gray-Box Uncertainty

Gray-box access assumes we have token probabilities but not raw logits.

---

## 4.1 Mean Log Probability

Longer sequences naturally have lower joint probability.
To normalize for length:

$$
\bar{\ell} = \frac{1}{N} \sum_{i=1}^{N} \log p_i
$$

This provides a **length-invariant confidence measure**.

We convert back:

$$
P = e^{\bar{\ell}}
$$

This represents the average per-token probability.

---

## 4.2 Self-Consistency Weighting

Behavioral stability is defined via repeated sampling:

$$
SelfConsistency = \frac{C_{max}}{N}
$$

Combined confidence:

$$
Confidence = SelfConsistency \cdot P
$$

Confidence must satisfy:

* High internal probability
* High behavioral agreement

Both reduce hallucination likelihood.

---

# 5. Black-Box Uncertainty

When only outputs are observable, we rely on behavioral statistics.

---

## 5.1 Self-Consistency

Repeated sampling gives multiple responses:

$$
Consistency = \frac{C_{max}}{N}
$$

Low consistency indicates instability.

---

## 5.2 Response Entropy

Let $K$ be the number of unique responses.

$$
H = - \sum_{i=1}^{K} p_i \log p_i
$$

Where $p_i$ is the frequency of response $i$.

High entropy → behavioral uncertainty
Low entropy → stable outputs

---

# 6. Semantic Consistency

Surface similarity is insufficient. Two responses may differ syntactically but convey identical meaning.

We embed responses into vector space:

$$
a, b \in \mathbb{R}^d
$$

---

## Cosine Similarity

$$
sim(a,b) = \frac{a \cdot b}{|a| |b|}
$$

* 1 → identical meaning
* 0 → unrelated
* -1 → opposite

Semantic consistency:

$$
S = \frac{1}{M} \sum_{k=1}^{M} sim_k
$$

Semantic uncertainty:

$$
U = 1 - S
$$

This captures **semantic drift**.

---

# 7. Final Hallucination Score

We integrate signals:

$$
Score = w_w \cdot White + w_g \cdot Gray + w_b \cdot Black
$$

Subject to:

$$
w_w + w_g + w_b = 1
$$

This weighted linear combination is:

* Interpretable
* Stable
* Optimizable

Weights can be tuned via validation.

---

# 8. Calibration Theory

Confidence should align with empirical accuracy.

---

## 8.1 Brier Score

$$
BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

Where:

* $p_i$ = predicted confidence
* $y_i \in {0,1}$

Lower is better.

---

## 8.2 Expected Calibration Error (ECE)

Partition predictions into bins $B_m$:

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{N}
\left| acc_m - conf_m \right|
$$

Where:

* $acc_m$ = empirical accuracy
* $conf_m$ = mean predicted confidence

High ECE → overconfidence or underconfidence.

---

# 9. Statistical Validation

---

## Pearson Correlation

$$
r = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$

Measures correlation between uncertainty and true error.

---

## AUROC

$$
AUROC = P(score_{hallucination} > score_{correct})
$$

* 0.5 → random
* 1.0 → perfect discrimination

---

# 10. Conceptual Decomposition

We propose:

$$
Hallucination = Overconfidence + Instability + SemanticDrift
$$

Where:

* Overconfidence → low entropy but wrong
* Instability → inconsistent outputs
* Semantic Drift → meaning inconsistency

---

# Final Insight

This framework treats hallucination not as a binary phenomenon, but as a **multi-dimensional epistemic failure**.

It integrates:

* Information theory
* Probabilistic modeling
* Behavioral sampling
* Representation learning
* Calibration theory

Thus providing a principled and extensible foundation for hallucination detection.

---

# 11. Practical Evaluation Protocol

This section explains **how to analyze the generated CSV metrics** and how researchers should interpret each uncertainty component.

Our exported CSV includes:

```
PROMPT
RESPONSE
Whitebox_Uncertainty
Graybox_Uncertainty
Blackbox_Uncertainty
Semantic_Uncertainty
Final_Score
Decision
Semantic_Similarity
Pearson_Corr
Spearman_Corr
AUROC
PR_AUC
Brier_Score
ECE
```

---

## 11.1 Row-Level Interpretation (Single Example Analysis)

For each prompt-response pair, inspect:

### 1️⃣ Whitebox_Uncertainty

High value → internal distribution is flat → epistemic uncertainty.

If:

* Low entropy
* But answer is wrong

This indicates **overconfidence failure**.

---

### 2️⃣ Graybox_Uncertainty

Measures average per-token log probability.

If:

* Low mean log probability → weak internal support
* High probability but wrong → miscalibration

---

### 3️⃣ Blackbox_Uncertainty

Based on repeated sampling.

If:

* Low self-consistency → behavioral instability
* High entropy across responses → model disagreement

Instability often correlates with hallucination risk.

---

### 4️⃣ Semantic_Uncertainty

Defined as:

```
U = 1 - S
```

Where S is average cosine similarity between response embeddings.

If:

* Low semantic similarity → meaning drift
* High drift → epistemic instability

---

### 5️⃣ Final_Score

Computed as weighted integration:

```
Score = w_w White + w_g Gray + w_b Black
```

Interpretation:

* Low score → reliable
* High score → hallucination risk

Decision column applies a threshold:

```
Decision = 1 if Score > tau else 0
```

---

# 12. Distribution-Level Analysis

Beyond individual examples, analyze metric distributions.

---

## 12.1 Histogram Inspection

Plot distributions of:

* Whitebox_Uncertainty
* Final_Score
* Semantic_Uncertainty

Look for:

* Bimodality (hallucinated vs correct clusters)
* Heavy tails (extreme uncertainty cases)

---

## 12.2 Hallucination Separation

If ground truth labels exist:

Compare:

```
mean_uncertainty(hallucinated)
vs
mean_uncertainty(correct)
```

A valid metric should satisfy:

```
mean_hallucinated > mean_correct
```

---

# 13. Correlation Analysis

## 13.1 Pearson Correlation

Measures linear relation between uncertainty and error.

Interpretation:

* r > 0.6 → strong linear relation
* 0.4–0.6 → moderate
* < 0.3 → weak

---

## 13.2 Spearman Correlation

Measures monotonic relationship.

Important when relationship is non-linear.

If:

Spearman > Pearson

→ relationship is monotonic but non-linear.

---

# 14. Classification Performance

## 14.1 AUROC

Measures probability that a hallucinated sample receives a higher score than a correct one.

Interpretation:

* 0.5 → random
* 0.7 → acceptable
* 0.8 → strong
* > 0.9 → excellent

Target: AUROC ≥ 0.8

---

## 14.2 PR-AUC

Important when dataset is imbalanced.

If hallucinations are rare, PR-AUC is more informative than AUROC.

---

# 15. Calibration Analysis

Hallucination is fundamentally an overconfidence problem.

Thus calibration metrics are critical.

---

## 15.1 Brier Score

```
BS = mean( (p - y)^2 )
```

Lower is better.

Measures probability accuracy.

---

## 15.2 Expected Calibration Error (ECE)

Measures mismatch between:

* Predicted confidence
* Empirical accuracy

Interpretation:

* < 0.05 → well calibrated
* 0.05–0.1 → acceptable
* > 0.1 → overconfident model

High AUROC but high ECE indicates:

> The model separates well but is miscalibrated.

---

# 16. Layer Ablation Study

To justify the multi-level framework, perform ablation:

| Configuration | AUROC |
| ------------- | ----- |
| White only    |       |
| Gray only     |       |
| Black only    |       |
| White + Gray  |       |
| All Layers    |       |

If:

```
AUROC(all) > AUROC(any single layer)
```

Then multi-level integration is validated.

This is essential for publication-level evaluation.

---

# 17. Failure Mode Analysis

Manually inspect:

* High confidence + wrong → overconfidence
* Low consistency + high entropy → instability
* Low semantic similarity → meaning drift

This helps decompose hallucination into:

```
Hallucination = Overconfidence + Instability + Semantic Drift
```

---

# 18. Reproducibility Notes

To ensure stable evaluation:

* Fix random seed for sampling
* Use consistent temperature
* Keep number of samples constant
* Normalize uncertainty scores if combining layers

---

# 19. Research Implications

This framework demonstrates that hallucination is not binary.

Instead, it is a structured epistemic failure across:

* Distributional space
* Sequence probability space
* Behavioral sampling space
* Representation space
* Calibration space

By integrating these layers, the system provides:

* Interpretable signals
* Modular extensibility
* Model-agnostic deployment compatibility
* Theoretical grounding in information theory and calibration theory

---



