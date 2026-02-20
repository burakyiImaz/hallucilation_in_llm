
# 🧠 Hallucination-in-LLM

**Unified White-Box, Gray-Box and Black-Box Hallucination Detection Framework**

---

# 📌 1. Problem Definition

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

🔍 4.1 White-Box Uncertainty
(1) Predictive Entropy

Token dağılımı için:

𝐻
(
𝑝
)
=
−
∑
𝑖
=
1
𝑉
𝑝
𝑖
log
⁡
𝑝
𝑖
H(p)=−
i=1
∑
V
	​

p
i
	​

logp
i
	​


Where:

V = vocabulary size

p_i = softmax probability

Higher entropy ⇒ higher uncertainty.

(2) Sequence Log Probability
log
⁡
𝑃
(
𝑦
)
=
∑
𝑡
=
1
𝑇
log
⁡
𝑃
(
𝑦
𝑡
∣
𝑦
<
𝑡
)
logP(y)=
t=1
∑
T
	​

logP(y
t
	​

∣y
<t
	​

)

Sequence probability:

𝑃
(
𝑦
)
=
𝑒
log
⁡
𝑃
(
𝑦
)
P(y)=e
logP(y)

Confidence:

𝐶
𝑜
𝑛
𝑓
𝑖
𝑑
𝑒
𝑛
𝑐
𝑒
=
𝑃
(
𝑦
)
Confidence=P(y)
🟡 4.2 Gray-Box Uncertainty
Mean Log Probability
ℓ
ˉ
=
1
𝑁
∑
𝑖
=
1
𝑁
log
⁡
𝑝
𝑖
ℓ
ˉ
=
N
1
	​

i=1
∑
N
	​

logp
i
	​


Converted to probability:

𝑃
=
𝑒
ℓ
ˉ
P=e
ℓ
ˉ

Final gray confidence:

𝐶
𝑜
𝑛
𝑓
𝑖
𝑑
𝑒
𝑛
𝑐
𝑒
=
𝑆
𝑒
𝑙
𝑓
𝐶
𝑜
𝑛
𝑠
𝑖
𝑠
𝑡
𝑒
𝑛
𝑐
𝑦
×
𝑃
Confidence=SelfConsistency×P
⚫ 4.3 Black-Box Uncertainty
Self-Consistency
𝐶
𝑜
𝑛
𝑠
𝑖
𝑠
𝑡
𝑒
𝑛
𝑐
𝑦
=
Most Common Response
Total Responses
Consistency=
Total Responses
Most Common Response
	​

Response Entropy
𝐻
=
−
∑
𝑖
=
1
𝐾
𝑝
𝑖
log
⁡
𝑝
𝑖
H=−
i=1
∑
K
	​

p
i
	​

logp
i
	​


Where:

K = number of unique responses

Higher entropy ⇒ higher disagreement.

🧠 4.4 Semantic Consistency

Cosine similarity:

𝑠
𝑖
𝑚
(
𝑎
,
𝑏
)
=
𝑎
⋅
𝑏
∣
∣
𝑎
∣
∣
 
∣
∣
𝑏
∣
∣
sim(a,b)=
∣∣a∣∣∣∣b∣∣
a⋅b
	​


Semantic consistency:

𝑆
=
1
𝑁
∑
𝑖
<
𝑗
𝑠
𝑖
𝑚
(
𝑒
𝑖
,
𝑒
𝑗
)
S=
N
1
	​

i<j
∑
	​

sim(e
i
	​

,e
j
	​

)

Uncertainty:

𝑈
=
1
−
𝑆
U=1−S
🎯 5. Final Hallucination Score
𝑆
𝑐
𝑜
𝑟
𝑒
=
𝑤
𝑤
⋅
𝑊
ℎ
𝑖
𝑡
𝑒
+
𝑤
𝑔
⋅
𝐺
𝑟
𝑎
𝑦
+
𝑤
𝑏
⋅
𝐵
𝑙
𝑎
𝑐
𝑘
Score=w
w
	​

⋅White+w
g
	​

⋅Gray+w
b
	​

⋅Black

Default weights:

White: 0.4
Gray : 0.3
Black: 0.3
📊 7. Calibration Metrics
Brier Score
𝐵
𝑆
=
1
𝑁
∑
𝑖
=
1
𝑁
(
𝑝
𝑖
−
𝑦
𝑖
)
2
BS=
N
1
	​

i=1
∑
N
	​

(p
i
	​

−y
i
	​

)
2

Lower is better.

Expected Calibration Error (ECE)
𝐸
𝐶
𝐸
=
∑
𝑚
=
1
𝑀
∣
𝐵
𝑚
∣
𝑁
∣
𝑎
𝑐
𝑐
(
𝐵
𝑚
)
−
𝑐
𝑜
𝑛
𝑓
(
𝐵
𝑚
)
∣
ECE=
m=1
∑
M
	​

N
∣B
m
	​

∣
	​

∣acc(B
m
	​

)−conf(B
m
	​

)∣

Measures calibration gap between confidence and accuracy.

📈 8. Statistical Metrics
Pearson Correlation
𝑟
=
𝑐
𝑜
𝑣
(
𝑋
,
𝑌
)
𝜎
𝑋
𝜎
𝑌
r=
σ
X
	​

σ
Y
	​

cov(X,Y)
	​

AUROC

Probability that a randomly chosen positive example is ranked higher than a randomly chosen negative one.
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


