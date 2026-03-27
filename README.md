````markdown
# Hallucination Detection in LLMs

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](#testing)

A modular and practical framework for detecting hallucinations in Large Language Models (LLMs) using **multi-level uncertainty signals** (black-box, gray-box, white-box), statistical metrics, and ground-truth similarity checks.

---

## 📸 Project Preview

> Add these images under `docs/assets/` for best GitHub rendering.

![Pipeline Overview](docs/assets/pipeline-overview.png)
![Example Evaluation Report](docs/assets/example-report.png)
![Metric Distribution](docs/assets/metric-distribution.png)

---

## ✨ Key Features

- **Multi-level uncertainty analysis**
  - Black-box: response consistency
  - Gray-box: log-probability signals
  - White-box: token/logit-level confidence
- **Statistical evaluation**
  - Pearson / Spearman correlation
  - AUROC / PR-AUC
  - Score distribution metrics
- **Text-ground-truth comparison**
  - String similarity
  - Keyword overlap
  - Combined semantic proxy score
- **Bilingual data support**
  - Turkish and English sample datasets
- **End-to-end evaluation pipeline**
  - Scoring, thresholding, and report generation

---

## 🧠 Why This Project?

LLMs can generate fluent but factually incorrect content.  
This repository provides a structured, extensible way to:

1. Quantify uncertainty
2. Compare outputs against reference answers
3. Produce interpretable hallucination risk levels

---

## 🏗️ Project Structure

```text
hallucilation_in_llm/
├── calibration.py
├── stat_metrics.py
├── test.ipynb
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── turkish_datasets.py
│   └── english_datasets.py
├── model/
│   ├── __init__.py
│   └── hf_model.py
├── uncertainty/
│   ├── __init__.py
│   ├── black_uncertainty.py
│   ├── graybox_uncertainty.py
│   ├── whitebox_uncertainty.py
│   └── semantic_uncertainty.py
├── evaluation/
│   ├── __init__.py
│   ├── aggregation.py
│   ├── evaluator.py
│   ├── ground_truth.py
│   ├── hallucination_score.py
│   ├── thresholds.py
│   └── report.py
├── decision/
│   ├── __init__.py
│   ├── final_score.py
│   └── hallucination_decider.py
├── pipeline/
│   ├── __init__.py
│   ├── runner.py
│   └── experiment.py
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/hallucilation_in_llm.git
cd hallucilation_in_llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from model import HFModel
from uncertainty import BlackBoxUncertainty, GrayBoxUncertainty, WhiteBoxUncertainty
from evaluation import HallucinationScore, HallucinationThresholds, EvaluationReport

model = HFModel(model_name="gpt2")
result = model.generate(
    prompt="What is the capital of Turkey?",
    max_new_tokens=30,
    num_samples=3
)

black = BlackBoxUncertainty(responses=result.responses).compute()
gray = GrayBoxUncertainty(
    responses=result.responses,
    log_probs=result.log_probs
).compute()
white = WhiteBoxUncertainty(
    scores=result.logits,
    token_ids=result.token_ids,
    text_responses=result.responses
).compute()

score = HallucinationScore(
    white_score=white.get("white_entropy", 0.0),
    gray_score=gray.get("gray_entropy", 0.0),
    black_score=black.get("black_entropy", 0.0)
).score()

risk = HallucinationThresholds().interpret(score)

report = EvaluationReport(
    hallucination_score=score,
    level=risk,
    white=white,
    gray=gray,
    black=black,
    language="en"
)

print("Risk:", risk)
print(report.to_json())
```

---

## 📊 Metrics Included

### Statistical
- Pearson correlation
- Spearman correlation
- AUROC
- PR-AUC
- Mean / Std / Min / Max / Median

### Semantic Proxy
- String similarity (`SequenceMatcher`)
- Keyword overlap ratio
- Combined similarity score
- Similarity variance across samples

---

## ∑ Mathematical Foundations

Let:
- \(x\): prompt
- \(R=\{r_1,\dots,r_n\}\): sampled responses
- \(g\): ground-truth answer
- \(y_i\in\{0,1\}\): hallucination label
- \(s_i\in[0,1]\): predicted hallucination score

### 1) Black-box uncertainty (response consistency)

Empirical response probability:
\[
\hat{p}(r)=\frac{\text{count}(r)}{n}
\]

Response entropy:
\[
U_{\text{black}}=-\sum_{r\in R_{\text{unique}}}\hat{p}(r)\log \hat{p}(r)
\]

Normalized form:
\[
\tilde{U}_{\text{black}}=\frac{U_{\text{black}}}{\log |R_{\text{unique}}|+\epsilon}
\]

High entropy implies low consistency.

---

### 2) Gray-box uncertainty (log-probability space)

For \(r_i=(w_1,\dots,w_T)\):
\[
\log P(r_i|x)=\sum_{t=1}^{T}\log P(w_t\mid w_{<t},x)
\]

Average token negative log-likelihood:
\[
\text{NLL}(r_i)=-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t\mid w_{<t},x)
\]

Perplexity:
\[
\text{PPL}(r_i)=\exp(\text{NLL}(r_i))
\]

Sequence confidence proxy:
\[
C_{\text{gray}}(r_i)=\exp(-\text{NLL}(r_i))
\]
\[
U_{\text{gray}}=1-\frac{1}{n}\sum_{i=1}^{n} C_{\text{gray}}(r_i)
\]

---

### 3) White-box uncertainty (token/logit space)

Given logits \(z_{t,k}\), softmax probabilities:
\[
p_t(k)=\frac{e^{z_{t,k}}}{\sum_j e^{z_{t,j}}}
\]

Token entropy:
\[
H_t=-\sum_k p_t(k)\log p_t(k)
\]

Mean token entropy:
\[
U_{\text{white}}=\frac{1}{T}\sum_{t=1}^{T}H_t
\]

Optional margin confidence:
\[
m_t=p_t(k_1)-p_t(k_2)
\]
where \(k_1,k_2\) are top-1 and top-2 tokens. Smaller \(m_t\) indicates higher uncertainty.

---

### 4) Ground-truth similarity

String similarity:
\[
S_{\text{str}}(r_i,g)\in[0,1]
\]

Keyword overlap:
\[
S_{\text{kw}}(r_i,g)=\frac{|K(r_i)\cap K(g)|}{|K(g)|+\epsilon}
\]

Averaged values:
\[
\bar{S}_{\text{str}}=\frac{1}{n}\sum_{i=1}^{n}S_{\text{str}}(r_i,g),\quad
\bar{S}_{\text{kw}}=\frac{1}{n}\sum_{i=1}^{n}S_{\text{kw}}(r_i,g)
\]

Combined semantic score:
\[
S_{\text{comb}}=\alpha \bar{S}_{\text{str}} + (1-\alpha)\bar{S}_{\text{kw}},\quad \alpha\in[0,1]
\]

Semantic variance (instability indicator):
\[
\mathrm{Var}_{\text{sem}}=\frac{1}{n}\sum_{i=1}^{n}
\left(S_{\text{str}}(r_i,g)-\bar{S}_{\text{str}}\right)^2
\]

---

### 5) Correlation and ranking quality

Pearson correlation:
\[
\rho_P=\frac{\sum_i (s_i-\bar{s})(y_i-\bar{y})}
{\sqrt{\sum_i(s_i-\bar{s})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}
\]

Spearman correlation:
\[
\rho_S=\rho_P(\mathrm{rank}(s),\mathrm{rank}(y))
\]

---

### 6) Classification quality curves

For threshold \(\tau\):
\[
\hat{y}_i(\tau)=\mathbb{1}[s_i\ge\tau]
\]

\[
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP+FN}},\quad
\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP+TN}}
\]
\[
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP+FP}},\quad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP+FN}}
\]

AUROC:
\[
\mathrm{AUROC}=\int_0^1 \mathrm{TPR}(u)\,d(\mathrm{FPR}(u))
\]

PR-AUC:
\[
\mathrm{PR\text{-}AUC}=\int_0^1 \mathrm{Precision}(r)\,d(\mathrm{Recall}(r))
\]

---

### 7) Final hallucination score fusion

Normalized fusion:
\[
H(x)=\sigma\!\left(
w_b\tilde{U}_{\text{black}}+
w_g\tilde{U}_{\text{gray}}+
w_w\tilde{U}_{\text{white}}-
w_s\tilde{S}_{\text{comb}}
\right)
\]
where:
- \(w_b,w_g,w_w,w_s\ge 0\)
- \(\sigma(\cdot)\): sigmoid or clipped mapping to \([0,1]\)

Risk mapping example:
\[
\text{LOW}: H<\tau_1,\quad
\text{MEDIUM}: \tau_1\le H<\tau_2,\quad
\text{HIGH}: \tau_2\le H<\tau_3,\quad
\text{CRITICAL}: H\ge\tau_3
\]

---

### 8) Calibration (optional but recommended)

Expected Calibration Error:
\[
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
\]

Brier Score:
\[
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(s_i-y_i)^2
\]

Lower ECE/Brier indicates better probabilistic reliability.

---

## 🧪 Testing

Run tests/notebook validation:

```bash
python -m pytest -q
```

or open:

- `test.ipynb` for full integration walkthrough.

---

## 🌍 Turkish & English Support

The project includes built-in loaders and samples for:

- Turkish Q&A / hallucination checks / math-style prompts
- English Q&A / TruthfulQA-style subsets

---

## 🖼️ Recommended Images to Add

Create these files for a professional README:

- `docs/assets/pipeline-overview.png` → architecture diagram
- `docs/assets/example-report.png` → sample JSON/report screenshot
- `docs/assets/metric-distribution.png` → score distribution chart

---

## 🛣️ Roadmap

- [ ] Add benchmark scripts (TruthfulQA, GSM8K-style sets)
- [ ] Add REST API demo for real-time scoring
- [ ] Add Docker support
- [ ] Add CI workflow for automated tests
- [ ] Add richer semantic similarity (embedding-based)

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License (or your preferred license).

---

## 📬 Contact

If you use this project in research or production, please open an issue or discussion for feedback.
````
