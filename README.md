Aşağıdaki metni **direkt README.md** içine yapıştırabilirsin.  
(Python örneği yok, tamamen matematiksel akış ve görsel odaklı.)

# Hallucination Detection in LLMs: A Mathematical Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Status](https://img.shields.io/badge/Status-Research%20Grade-success.svg)](#)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

A mathematically grounded framework for quantifying hallucination risk in Large Language Models (LLMs) via multi-level uncertainty signals and statistical validation.

---

## 🖼 Visual Summary

![System Overview](docs/assets/01_system_overview.png)  
![Uncertainty Layers](docs/assets/02_uncertainty_layers.png)  
![Score Fusion](docs/assets/03_score_fusion.png)  
![Thresholding & Risk Bands](docs/assets/04_thresholding.png)  
![ROC and PR Curves](docs/assets/05_roc_pr.png)  
![Calibration Plot](docs/assets/06_calibration.png)

---

## 1) Problem Definition

Given:
- a prompt \(x\),
- a set of model outputs \(R=\{r_1,\dots,r_n\}\),
- optional token-level distributions/logits,
- optional ground-truth answer \(g\),

we estimate a hallucination-risk function:

$$
H(x)\in[0,1]
$$

where:
- \(H(x)\approx 0\): low hallucination risk,
- \(H(x)\approx 1\): high hallucination risk.

---

## 2) End-to-End Mathematical Flow

**Input** \(\rightarrow\) **Uncertainty extraction** \(\rightarrow\) **Normalization** \(\rightarrow\) **Fusion** \(\rightarrow\) **Calibration** \(\rightarrow\) **Risk decision**

Formally:

$$
x \mapsto \Big(U_{\text{black}},U_{\text{gray}},U_{\text{white}},S_{\text{gt}}\Big)
\mapsto \tilde{\mathbf{z}}
\mapsto H(x)
\mapsto \hat{H}(x)
\mapsto \text{Risk Level}
$$

where:
- \(U_{\text{black}}\): response-level inconsistency,
- \(U_{\text{gray}}\): probability-space uncertainty,
- \(U_{\text{white}}\): token/logit uncertainty,
- \(S_{\text{gt}}\): ground-truth agreement score.

---

## 3) Signal Layer I — Black-Box Uncertainty

No internals required; only response samples.

Let empirical response frequency be:

$$
\hat{p}(r)=\frac{\text{count}(r)}{n}
$$

Entropy-based inconsistency:

$$
U_{\text{black}}=-\sum_{r\in R_{\text{unique}}}\hat{p}(r)\log\hat{p}(r)
$$

Normalized entropy:

$$
\tilde{U}_{\text{black}}=
\frac{U_{\text{black}}}{\log|R_{\text{unique}}|+\varepsilon}
$$

Interpretation:
- low entropy \(\Rightarrow\) consistent answers,
- high entropy \(\Rightarrow\) unstable generation behavior.

---

## 4) Signal Layer II — Gray-Box Uncertainty

Uses token log-probabilities (without full internal state).

For response \(r_i=(w_1,\dots,w_T)\):

$$
\log P(r_i\mid x)=\sum_{t=1}^T \log P(w_t\mid w_{<t},x)
$$

Average negative log-likelihood:

$$
\text{NLL}(r_i)=-\frac{1}{T}\sum_{t=1}^T\log P(w_t\mid w_{<t},x)
$$

Perplexity:

$$
\text{PPL}(r_i)=e^{\text{NLL}(r_i)}
$$

Confidence proxy:

$$
C_{\text{gray}}(r_i)=e^{-\text{NLL}(r_i)}
$$

Aggregate uncertainty:

$$
U_{\text{gray}}=1-\frac{1}{n}\sum_{i=1}^n C_{\text{gray}}(r_i)
$$

---

## 5) Signal Layer III — White-Box Uncertainty

Uses logits/probability vectors directly.

With logits \(z_{t,k}\), token distribution:

$$
p_t(k)=\frac{e^{z_{t,k}}}{\sum_j e^{z_{t,j}}}
$$

Token entropy:

$$
H_t=-\sum_k p_t(k)\log p_t(k)
$$

Sequence uncertainty:

$$
U_{\text{white}}=\frac{1}{T}\sum_{t=1}^T H_t
$$

Optional margin confidence:

$$
m_t=p_t(k_{(1)})-p_t(k_{(2)})
$$

where \(k_{(1)},k_{(2)}\) are top-1 and top-2 classes. Smaller \(m_t\) implies higher ambiguity.

---

## 6) Ground-Truth Agreement Modeling

For each response \(r_i\) and reference \(g\):

- structural/textual similarity \(S_{\text{str}}(r_i,g)\in[0,1]\),
- keyword overlap score \(S_{\text{kw}}(r_i,g)\in[0,1]\).

Mean agreement:

$$
\bar{S}_{\text{str}}=\frac{1}{n}\sum_{i=1}^n S_{\text{str}}(r_i,g),
\quad
\bar{S}_{\text{kw}}=\frac{1}{n}\sum_{i=1}^n S_{\text{kw}}(r_i,g)
$$

Combined reference agreement:

$$
S_{\text{gt}}=\alpha\bar{S}_{\text{str}}+(1-\alpha)\bar{S}_{\text{kw}},\quad \alpha\in[0,1]
$$

Stability term (variance):

$$
\mathrm{Var}_{\text{sem}}=
\frac{1}{n}\sum_{i=1}^n
\left(S_{\text{str}}(r_i,g)-\bar{S}_{\text{str}}\right)^2
$$

---

## 7) Normalization and Feature Vector

Define feature vector:

$$
\mathbf{z}=
\big(U_{\text{black}},U_{\text{gray}},U_{\text{white}},S_{\text{gt}},\mathrm{Var}_{\text{sem}}\big)
$$

Normalized vector:

$$
\tilde{\mathbf{z}}=\mathcal{N}(\mathbf{z})
$$

where \(\mathcal{N}\) may be min-max, z-score, or robust scaling.

---

## 8) Fusion Function (Hallucination Score)

A monotonic fusion:

$$
H(x)=\sigma\!\left(
w_b\tilde{U}_{\text{black}}+
w_g\tilde{U}_{\text{gray}}+
w_w\tilde{U}_{\text{white}}-
w_s\tilde{S}_{\text{gt}}+
w_v\widetilde{\mathrm{Var}}_{\text{sem}}
+b
\right)
$$

Constraints:

$$
w_b,w_g,w_w,w_s,w_v\ge 0
$$

\(\sigma(\cdot)\) is typically sigmoid:

$$
\sigma(t)=\frac{1}{1+e^{-t}}
$$

---

## 9) Decision Rule and Risk Bands

Given thresholds \(0\le\tau_1<\tau_2<\tau_3\le1\):

- **LOW** if \(H(x)<\tau_1\)
- **MEDIUM** if \(\tau_1\le H(x)<\tau_2\)
- **HIGH** if \(\tau_2\le H(x)<\tau_3\)
- **CRITICAL** if \(H(x)\ge\tau_3\)

This yields interpretable operational decisions.

---

## 10) Statistical Evaluation Criteria

Let \((s_i,y_i)\) be predicted score and binary label.

Pearson correlation:

$$
\rho_P=
\frac{\sum_i (s_i-\bar{s})(y_i-\bar{y})}
{\sqrt{\sum_i (s_i-\bar{s})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
$$

Spearman correlation (rank-based):

$$
\rho_S=\rho_P(\mathrm{rank}(s),\mathrm{rank}(y))
$$

ROC quantities:

$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP+FN}},
\quad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP+TN}}
$$

Precision-recall quantities:

$$
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP+FP}},
\quad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP+FN}}
$$

Areas:

$$
\mathrm{AUROC}=\int_0^1 \mathrm{TPR}(u)\,d(\mathrm{FPR}(u)),
\quad
\mathrm{PR\text{-}AUC}=\int_0^1 \mathrm{Precision}(r)\,d(\mathrm{Recall}(r))
$$

---

## 11) Calibration Quality

Calibration checks whether predicted risk matches empirical frequency.

Expected Calibration Error:

$$
\mathrm{ECE}=
\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
$$

Brier score:

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(s_i-y_i)^2
$$

Lower ECE and Brier indicate more trustworthy risk probabilities.

---

## 12) Sensitivity and Ablation Logic

To quantify each signal’s contribution:

$$
\Delta_j = \mathcal{M}(\text{all features})-\mathcal{M}(\text{without feature }j)
$$

where \(\mathcal{M}\) can be AUROC, PR-AUC, or calibration quality.  
Large \(\Delta_j\) implies feature \(j\) has strong explanatory power.

---

## 13) Computational Profile (Theoretical)

For \(n\) responses, average token length \(T\), vocabulary size \(|V|\):

- Black-box entropy: \(O(n)\)
- Gray-box NLL aggregation: \(O(nT)\)
- White-box entropy (full distribution): \(O(nT|V|)\)
- Similarity aggregation: depends on string/token operations, typically \(O(nT)\) to \(O(nT\log T)\)

---

## 14) Practical Interpretation Guide

- High \(U_{\text{black}}\) + high \(U_{\text{white}}\): unstable and uncertain generation.
- Low \(S_{\text{gt}}\): poor factual agreement with reference.
- High \(H(x)\) + poor calibration: strong candidate for review/rejection.
- Medium \(H(x)\): prefer fallback retrieval/tool verification before final output.

---

## 15) Recommended Visual Assets (for a Rich GitHub Page)

Add these images to `docs/assets/`:

1. `01_system_overview.png` — full pipeline diagram  
2. `02_uncertainty_layers.png` — black/gray/white comparison  
3. `03_score_fusion.png` — weighted fusion illustration  
4. `04_thresholding.png` — risk-band thresholds  
5. `05_roc_pr.png` — ROC and PR curves  
6. `06_calibration.png` — reliability diagram + ECE

---

## License

MIT

---

## Citation

If this framework is used in research or production, please cite the repository and include the project URL.
