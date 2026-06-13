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


---

## Issue #2: Exploratory Data Analysis (EDA) & Feature Engineering

### 1. Finance & Risk Assessment: Behavioral Risk Proxies & The "One-to-Many" Problem
* **Objective:** Translate raw transactional data (e.g., monthly balance snapshots, missed installments) into behavioral risk proxies.
* **Core Concept:** * **The "One-to-Many" Problem:** Relational SQL databases store a client's current status (one row) separately from their history (many rows). If we join them directly, we cause a "row explosion" that creates duplicate data and breaks the machine learning model.
  * **Behavioral Risk Proxies:** Instead of feeding raw transaction logs into the model, we use our Feature Engineering pipeline to aggregate a client's history into a single, meaningful mathematical summary (a proxy). This transforms the fragmented SQL data into a single "Feature Vector" (a flat row of numbers) that the algorithm can process.

### 2. Machine Learning: Interpretable Feature Engineering vs. Dimensionality Reduction
* **Objective:** Engineer advanced predictive features without violating regulatory interpretability requirements.
* **Core Concept:**
  * **The PCA Trap:** While techniques like Principal Component Analysis (PCA) are great for compressing data, they create unexplainable outputs (e.g., "Principal Component 1"). If a loan is denied based on PCA, we cannot legally explain the decision to the customer, violating GDPR/CCAR.
  * **Advanced White-Box Proxies:** Instead of black-box dimensionality reduction, we use advanced, interpretable math. Examples include calculating **Trend Slopes** (using 1D linear regression to see if debt is accelerating or decelerating) or **Time-Decay Aggregations** (using Exponential Moving Averages to heavily weight recent late payments while forgiving older ones). This keeps the model both highly accurate and perfectly explainable.


---

**"Apply transparent feature selection (RFE) and monotonic binning (WoE/IV) to handle massive feature spaces and non-linearities without losing regulatory interpretability."**

**My Explanation:**
Since we cannot use PCA (because it creates unreadable "black box" components), we must reduce our 500+ features using methods that regulators love:

**1. Recursive Feature Elimination (RFE)**
Instead of mashing 10 variables into 1 unreadable variable (like PCA does), RFE acts like a ruthless editor. It trains a model, looks at all 500 features, and drops the weakest one. Then it trains again on 499, and drops the weakest. It repeats this until you are left with only the "Top 30" most powerful, *original*, readable features. You keep the transparency, but lose the noise.

**2. Weight of Evidence (WoE) & Information Value (IV)**
This is the gold standard in banking. Earlier, we discussed how linear models struggle with "tipping points" (like the U-shape risk of having 0 vs. 3 vs. 15 credit cards).

* **WoE** solves this by grouping the data into "bins" (e.g., Bin 1: 0 cards, Bin 2: 1-4 cards, Bin 3: 5+ cards). It then calculates the exact historical risk (the Weight of Evidence) for *just that bin*.
* We replace the raw number of cards with the WoE score. The Logistic Regression model can now easily draw a straight line through these scores, allowing a purely linear model to perfectly capture complex, non-linear human behavior!
* **IV (Information Value)** is just the total predictive power of that feature. If a feature's IV is too low, we drop it before training.

By combining RFE (to drop useless features) and WoE (to handle non-linear tipping points), you get a model that is as powerful as a complex Neural Network but as transparent as basic arithmetic.

---

## Issue #3: The "Black Box" Ensembles & Post-Hoc Interpretability

### 1. Finance & Risk Assessment: The Accuracy vs. Explainability Trade-Off
* **Objective:** Identify the tension between maximizing predictive accuracy to save the bank money versus the compliance risk of deploying models that management cannot fully explain.
* **Core Concept:**
  * **The Conflict:** Banks face a fundamental trade-off — regulations (GDPR, ECOA, SR 11-7) require that every credit decision be explainable to the customer and to auditors. Logistic Regression achieves this perfectly through its coefficients, but it lacks the accuracy to capture the non-linear complexity of human financial behavior.
  * **Why Banks Want Black Boxes:** Ensemble models (XGBoost, LightGBM) and Neural Networks achieve significantly higher AUC by modeling non-linear relationships and feature interactions through hundreds of sequential decision trees. Even a few percentage points of AUC improvement translates to millions saved through fewer surprise defaults and better capital allocation.
  * **Why Regulators Reject Them:** These models have thousands of internal split rules, making it impossible to provide a simple, legally defensible explanation like "your debt-to-income coefficient was -1.8." They also risk learning proxy discrimination (e.g., using zip code as a hidden proxy for race).
  * **The SHAP Compromise:** In 2017, Lundberg & Lee adapted **Shapley values** (a concept from cooperative game theory, 1953) to machine learning. SHAP treats each feature as a "player" and calculates its fair contribution to the prediction by examining all permutations of feature subsets with and without that specific feature.
  * **KernelSHAP vs. TreeSHAP:** KernelSHAP is the model-agnostic (generalized) version — it works on any model but is computationally expensive and produces approximations. TreeSHAP is optimized for tree-based models (like our XGBoost/LightGBM) and computes exact Shapley values efficiently.
  * **Global vs. Local Explanations:** SHAP provides both global feature importance (which features matter most across all clients) and local explanations (which features drove the decision for one specific client).
  * **Critical Limitation:** SHAP is a **post-hoc** explanation — it is applied *after* the model is built. Unlike Logistic Regression where explainability is **intrinsic** (built into the math), SHAP is an external approximation bolted onto an opaque model. Some regulators accept it; others do not. This unresolved debate is why architectures like the Double-Tree (Issue #4) aim to achieve both accuracy and intrinsic transparency.

