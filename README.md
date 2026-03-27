# Hallucination Detection in LLMs  
### A Mathematical, Multi-Signal Reliability Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#)  
[![Focus](https://img.shields.io/badge/Focus-Hallucination%20Risk-orange.svg)](#)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

This repository presents a mathematically grounded framework for measuring hallucination risk in Large Language Models (LLMs) via **black-box**, **gray-box**, and **white-box** uncertainty signals, then mapping them into a calibrated risk score.

---

## Visual Overview

> Place these files under `docs/assets/` for full GitHub rendering.

![System Overview](docs/assets/01_system_overview.png)  
![Uncertainty Layers](docs/assets/02_uncertainty_layers.png)  
![Fusion and Thresholding](docs/assets/03_fusion_thresholds.png)  
![ROC and PR Curves](docs/assets/04_roc_pr.png)  
![Calibration Diagram](docs/assets/05_calibration.png)

---

## End-to-End Flow

```mermaid
flowchart LR
    A[Prompt x] --> B[Response Sampling R = r1...rn]
    B --> C1[Black-Box Signal U_black]
    B --> C2[Gray-Box Signal U_gray]
    B --> C3[White-Box Signal U_white]
    B --> C4[Ground-Truth Agreement S_gt]
    C1 --> D[Normalization]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> E[Fusion Function H(x)]
    E --> F[Calibration H_hat(x)]
    F --> G[Risk Bands: LOW MEDIUM HIGH CRITICAL]
```

---

## 1) Formal Problem Statement

Given:
- prompt \(x\),
- sampled responses \(R=\{r_1,\ldots,r_n\}\),
- optional token probability/logit traces,
- optional ground truth \(g\),

estimate hallucination risk:
\[
H(x)\in[0,1]
\]
with:
- \(H(x)\approx 0\): reliable output behavior,
- \(H(x)\approx 1\): high hallucination likelihood.

---

## 2) Signal Layer I: Black-Box Uncertainty

Black-box uses only generated outputs.

Empirical response distribution:
\[
\hat{p}(r)=\frac{\mathrm{count}(r)}{n}
\]

Response entropy:
\[
U_{\mathrm{black}}=-\sum_{r\in R_{\mathrm{unique}}}\hat{p}(r)\log\hat{p}(r)
\]

Normalized entropy:
\[
\tilde{U}_{\mathrm{black}}=
\frac{U_{\mathrm{black}}}{\log\!\big(|R_{\mathrm{unique}}|\big)+\varepsilon}
\]

Interpretation:
- low entropy: stable responses,
- high entropy: inconsistent generations and higher uncertainty.

---

## 3) Signal Layer II: Gray-Box Uncertainty

Gray-box uses token-level log-probabilities without full model internals.

For \(r_i=(w_1,\ldots,w_T)\):
\[
\log P(r_i\mid x)=\sum_{t=1}^{T}\log P\!\left(w_t\mid w_{1:t-1},x\right)
\]

Average token negative log-likelihood:
\[
\mathrm{NLL}(r_i)=
-\frac{1}{T}\sum_{t=1}^{T}\log P\!\left(w_t\mid w_{1:t-1},x\right)
\]

Perplexity:
\[
\mathrm{PPL}(r_i)=\exp\!\left(\mathrm{NLL}(r_i)\right)
\]

Confidence proxy:
\[
C_{\mathrm{gray}}(r_i)=\exp\!\left(-\mathrm{NLL}(r_i)\right)
\]

Aggregate gray uncertainty:
\[
U_{\mathrm{gray}}=
1-\frac{1}{n}\sum_{i=1}^{n}C_{\mathrm{gray}}(r_i)
\]

---

## 4) Signal Layer III: White-Box Uncertainty

White-box uses logits/probability vectors directly.

Given logits \(z_{t,k}\), token distribution:
\[
p_t(k)=\frac{\exp(z_{t,k})}{\sum_j \exp(z_{t,j})}
\]

Token entropy:
\[
H_t=-\sum_k p_t(k)\log p_t(k)
\]

Sequence-level white uncertainty:
\[
U_{\mathrm{white}}=\frac{1}{T}\sum_{t=1}^{T}H_t
\]

Optional margin confidence:
\[
m_t=p_t(k_{(1)})-p_t(k_{(2)})
\]
where \(k_{(1)}\) and \(k_{(2)}\) are top-1 and top-2 token indices.

---

## 5) Ground-Truth Agreement Modeling

For response \(r_i\) and reference \(g\):
- structural similarity \(S_{\mathrm{str}}(r_i,g)\in[0,1]\),
- keyword overlap \(S_{\mathrm{kw}}(r_i,g)\in[0,1]\).

Mean similarities:
\[
\bar{S}_{\mathrm{str}}=\frac{1}{n}\sum_{i=1}^{n}S_{\mathrm{str}}(r_i,g),\quad
\bar{S}_{\mathrm{kw}}=\frac{1}{n}\sum_{i=1}^{n}S_{\mathrm{kw}}(r_i,g)
\]

Combined agreement:
\[
S_{\mathrm{gt}}=\alpha\bar{S}_{\mathrm{str}}+(1-\alpha)\bar{S}_{\mathrm{kw}},\quad \alpha\in[0,1]
\]

Stability proxy:
\[
\mathrm{Var}_{\mathrm{sem}}=
\frac{1}{n}\sum_{i=1}^{n}
\left(S_{\mathrm{str}}(r_i,g)-\bar{S}_{\mathrm{str}}\right)^2
\]

---

## 6) Normalization and Feature Vector

Signal vector:
\[
\mathbf{z}=
\left(
U_{\mathrm{black}},
U_{\mathrm{gray}},
U_{\mathrm{white}},
S_{\mathrm{gt}},
\mathrm{Var}_{\mathrm{sem}}
\right)
\]

Normalized features:
\[
\tilde{\mathbf{z}}=\mathcal{N}(\mathbf{z})
\]
where \(\mathcal{N}\) can be min-max, z-score, or robust scaling.

---

## 7) Fusion Function (Hallucination Score)

Weighted fusion:
\[
H(x)=\sigma\!\left(
w_b\tilde{U}_{\mathrm{black}}+
w_g\tilde{U}_{\mathrm{gray}}+
w_w\tilde{U}_{\mathrm{white}}-
w_s\tilde{S}_{\mathrm{gt}}+
w_v\widetilde{\mathrm{Var}}_{\mathrm{sem}}+b
\right)
\]

with:
\[
w_b,w_g,w_w,w_s,w_v\ge 0
\]
and sigmoid:
\[
\sigma(t)=\frac{1}{1+\exp(-t)}
\]

---

## 8) Risk Bands and Decision Rule

Given thresholds:
\[
0\le\tau_1<\tau_2<\tau_3\le 1
\]

Decision mapping:
- **LOW** if \(H(x)<\tau_1\)
- **MEDIUM** if \(\tau_1\le H(x)<\tau_2\)
- **HIGH** if \(\tau_2\le H(x)<\tau_3\)
- **CRITICAL** if \(H(x)\ge\tau_3\)

---

## 9) Statistical Validation Metrics

For predicted scores \(s_i\) and labels \(y_i\in\{0,1\}\):

Pearson:
\[
\rho_P=
\frac{\sum_i (s_i-\bar{s})(y_i-\bar{y})}
{\sqrt{\sum_i (s_i-\bar{s})^2}\sqrt{\sum_i (y_i-\bar{y})^2}}
\]

Spearman:
\[
\rho_S=\rho_P\big(\mathrm{rank}(s),\mathrm{rank}(y)\big)
\]

Thresholded predictions:
\[
\hat{y}_i(\tau)=\mathbf{1}[s_i\ge \tau]
\]

Rates:
\[
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
\]
\[
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
\]

Areas:
\[
\mathrm{AUROC}=\int_0^1 \mathrm{TPR}(u)\,d(\mathrm{FPR}(u))
\]
\[
\mathrm{PRAUC}=\int_0^1 \mathrm{Precision}(r)\,d(\mathrm{Recall}(r))
\]

---

## 10) Calibration Quality

Expected Calibration Error:
\[
\mathrm{ECE}=
\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
\]

Brier score:
\[
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(s_i-y_i)^2
\]

Lower ECE and Brier indicate more reliable risk probabilities.

---

## 11) Recommended Ablation Protocol

To measure signal importance, remove one feature family at a time:

\[
\Delta_j=\mathcal{M}(\text{full})-\mathcal{M}(\text{without signal }j)
\]

where \(\mathcal{M}\in\{\mathrm{AUROC},\mathrm{PRAUC},-\mathrm{ECE}\}\).

Large \(\Delta_j\) implies high contribution of signal \(j\).

---

## 12) Complexity Profile (High-Level)

Let:
- \(n\): number of responses,
- \(T\): average response length,
- \(|V|\): vocabulary size.

Approximate costs:
- black-box entropy: \(O(n)\),
- gray-box NLL aggregation: \(O(nT)\),
- white-box entropy: \(O(nT|V|)\),
- similarity aggregation: typically \(O(nT)\) to \(O(nT\log T)\).

---

## 13) Practical Interpretation

- High \(U_{\mathrm{black}}\) + high \(U_{\mathrm{white}}\): unstable and uncertain decoding.
- Low \(S_{\mathrm{gt}}\): weak factual alignment with reference.
- High \(H(x)\): prioritize retrieval/tool-checking or human review.
- Medium \(H(x)\): use fallback verification before final output.

---

## 14) Visual Assets Checklist

For a rich repository page, include:

- `docs/assets/01_system_overview.png`
- `docs/assets/02_uncertainty_layers.png`
- `docs/assets/03_fusion_thresholds.png`
- `docs/assets/04_roc_pr.png`
- `docs/assets/05_calibration.png`

---

## 15) Repository Scope

Current scope includes:
- multi-level uncertainty decomposition,
- score fusion and risk interpretation,
- correlation/discrimination/calibration evaluation,
- multilingual orientation (Turkish + English settings).

---

## License

MIT

---

## Citation

If used in research or production, please cite this repository and link the project URL.
````
