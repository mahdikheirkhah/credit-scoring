# Project Learning Objectives: Credit Scoring & Econometric ML

## Issue #1: Repository Setup & The Econometric Baseline (Logistic Regression)

### 1. Finance & Risk Assessment: Interpretable Econometric Models
* **Objective:** Understand why the financial industry relies on highly interpretable econometric models to comply with regulations.
* **Core Concept:** * As a risk manager, it is not enough for a model to just be highly accurate (a "black box"). We must be able to mathematically justify and explain the exact reasons for a credit decision.
  * **White-Box Models:** Traditional models like Logistic Regression give exact weights (coefficients) to every factor, allowing us to explain exactly why a decision was made.
  * **Regulations:** This transparency is required to comply with laws like **GDPR** in Europe (which gives citizens the "right to explanation" for automated decisions) and **CCAR** in the US (which requires banks to prove their models are mathematically sound to regulators).

---

### 2. Machine Learning: Reproducible Pipelines & Linear Limitations
* **Objective:** Establish a reproducible pipeline and evaluate the limitations of linear models when dealing with complex, non-linear financial behaviors.
* **Core Concept:** * **Reproducible Pipeline:** An automated "assembly line" for the code. From reading data to outputting the model, every step must run in the exact same sequence. This ensures that if we change a parameter, the whole pipeline aligns, eliminating human error and proving to auditors that our results are 100% reproducible.
  * **Limitations of Linear Models:** Human financial behavior is highly non-linear and contains complex feature interactions. A simple straight line cannot properly separate or cluster these complex behaviors. While linear models offer great transparency, they fail to capture the true, complicated nature of debt and risk.