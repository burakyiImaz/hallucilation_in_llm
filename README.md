
# 🧠 Hallucination-in-LLM

**Unified White-Box, Gray-Box and Black-Box Hallucination Detection Framework**

---

## 📌 1. Problem Definition

Large Language Models (LLMs) often generate fluent but factually incorrect answers.
This phenomenon is known as:

> **Hallucination**

Hallucinations are dangerous in:

* Healthcare
* Legal systems
* Finance
* Education
* Autonomous agents

This project builds a **multi-perspective uncertainty evaluation pipeline** to detect hallucinations using:

* 🔍 White-box signals (logits, entropy)
* 🟡 Gray-box signals (token probabilities)
* ⚫ Black-box signals (response diversity)
* 🧠 Semantic consistency (embedding similarity)
* 📊 Calibration metrics
* 📈 Statistical reliability analysis

---

# 🏗️ 2. Project Architecture

```
User Prompt
    ↓
HFModel.generate()
    ↓
ModelOutput
    ↓
Uncertainty Modules
    ├── WhiteBoxUncertainty
    ├── GrayBoxUncertainty
    ├── BlackBoxUncertainty
    ├── SemanticUncertainty
    ↓
Evaluator
    ↓
FinalScore
    ↓
HallucinationDecider
    ↓
Evaluation Report
```

---

# 📂 3. Folder Structure

```
hallucilation_in_llm/
│
├── model/              # HFModel and output containers
├── uncertainty/        # White, Gray, Black, Semantic uncertainty
├── evaluation/         # Scoring, thresholds, reports
├── decision/           # Final decision logic
├── pipeline/           # Runner & experiment orchestration
├── visualization/      # Diagrams & plots
│
├── calibration.py      # Calibration metrics (Brier, ECE)
├── stat_metrics.py     # Correlation, AUROC, PR-AUC
│
├── pipeline_results_en.csv
├── pipeline_results_tr.csv
└── test.ipynb
```

---

# 🧠 4. Uncertainty Modules (Mathematical Foundations)

---

# 🔍 4.1 White-Box Uncertainty

Uses internal model logits.

### (1) Predictive Entropy

For token distribution:

[
H(p) = - \sum_{i=1}^{V} p_i \log p_i
]

Where:

* ( V ) = vocabulary size
* ( p_i ) = softmax probability

Higher entropy → more uncertainty.

---

### (2) Sequence Log Probability

[
\log P(y) = \sum_{t=1}^{T} \log P(y_t | y_{<t})
]

Sequence probability:

[
P(y) = e^{\log P(y)}
]

Confidence:

[
Confidence = P(y)
]

---

# 🟡 4.2 Gray-Box Uncertainty

Uses token log-probabilities (no full logits).

### Mean Log Probability

[
\bar{\ell} = \frac{1}{N} \sum_{i=1}^{N} \log p_i
]

Converted to probability:

[
P = e^{\bar{\ell}}
]

Final gray confidence:

[
Confidence = SelfConsistency \times P
]

---

# ⚫ 4.3 Black-Box Uncertainty

Uses only generated text.

### Self-Consistency

[
Consistency = \frac{Most\ Common\ Response}{Total\ Responses}
]

---

### Response Entropy

[
H = - \sum_{i=1}^{K} p_i \log p_i
]

Where:

* (K) = unique responses

Higher entropy → more disagreement.

---

# 🧠 4.4 Semantic Consistency

Uses sentence embeddings.

Cosine similarity:

[
sim(a,b) = \frac{a \cdot b}{||a|| ||b||}
]

Semantic consistency:

[
S = \frac{1}{N} \sum_{i<j} sim(e_i, e_j)
]

Uncertainty:

[
U = 1 - S
]

---

# 🎯 5. Final Hallucination Score

[
Score =
w_w \cdot White +
w_g \cdot Gray +
w_b \cdot Black
]

Default weights:

```
White: 0.4
Gray : 0.3
Black: 0.3
```

---

# 🚦 6. Risk Interpretation

| Score | Risk     |
| ----- | -------- |
| < 0.3 | LOW      |
| < 0.6 | MEDIUM   |
| < 0.8 | HIGH     |
| ≥ 0.8 | CRITICAL |

---

# 📊 7. Calibration Metrics

---

## Brier Score

[
BS = \frac{1}{N} \sum (p_i - y_i)^2
]

Lower is better.

---

## Expected Calibration Error (ECE)

[
ECE = \sum_{m=1}^{M}
\frac{|B_m|}{N}
|acc(B_m) - conf(B_m)|
]

Measures calibration gap.

---

# 📈 8. Statistical Metrics

### Pearson Correlation

[
r = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
]

### Spearman Correlation

Rank-based correlation.

### AUROC

Probability model ranks positive higher than negative.

### PR-AUC

Area under Precision-Recall curve.

---

# 🚀 9. How to Run the Pipeline

---

## Step 1 — Install Requirements

```bash
pip install torch transformers sentence-transformers numpy scipy scikit-learn matplotlib seaborn networkx
```

---

## Step 2 — Example Usage

```python
from model.hf_model import HFModel
from pipeline.runner import PipelineRunner
from evaluation.evaluator import Evaluator
from decision.final_score import FinalScore
from decision.hallucination_decider import HallucinationDecider
from uncertainty.black_uncertainty import BlackBoxUncertainty

model = HFModel("gpt2")

uncertainty_modules = {
    "black": lambda output: BlackBoxUncertainty(output.responses)
}

evaluator = Evaluator(FinalScore())
decider = HallucinationDecider({"hallucination": 0.6})

runner = PipelineRunner(model, uncertainty_modules, evaluator, decider)

result = runner.run("What is the capital of France?")
print(result)
```

---

# 🧪 10. Running Experiments

Use:

```
pipeline/experiment.py
```

Provide:

* Prompt list
* Ground truth labels

Then evaluate with:

* `stat_metrics.py`
* `calibration.py`

---

# 📊 11. Visualization

Run:

```python
from visualization.visualization_manager import VisualizationManager
VisualizationManager().generate_all()
```

Outputs:

* Class diagram
* Token flow
* Logit heatmaps
* API simulations

---

# 🎓 12. Academic Value

This project demonstrates:

* Multi-level uncertainty modeling
* Probabilistic calibration
* Statistical validation
* Risk-aware LLM deployment design
* Modular research-grade architecture

It can be extended into:

* Research paper
* TÜBİTAK project
* Master's thesis
* Production risk system

---

# 🧩 13. Future Extensions

* Bayesian ensembling
* Monte Carlo dropout
* Retrieval verification
* Knowledge-grounded validation
* RLHF-aware calibration
* Adaptive thresholding

---

# 👤 Author

Burak Yılmaz
AI & Uncertainty Research
Data Science & LLM Systems

---


