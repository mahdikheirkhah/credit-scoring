# 🏦 Econometric ML for Credit Risk Scoring

An end-to-end, serverless machine learning pipeline and interactive auditing dashboard designed to bridge the gap between high-accuracy predictive modeling and strict financial regulatory compliance.

## 📑 Table of Contents
1. [Project Goal](#-project-goal)
2. [Our Solution](#-our-solution)
3. [Methodology & Steps Taken](#-methodology--steps-taken)
4. [Data & Feature Dictionary](#-data--feature-dictionary)
5. [System Requirements](#-system-requirements)
6. [How to Run the Project](#-how-to-run-the-project)
7. [Architecture & Tech Stack](#-architecture--tech-stack)

---

## 🎯 Project Goal
Modern financial institutions face a critical dilemma: advanced machine learning models (like LightGBM and XGBoost) offer superior predictive accuracy for credit default risk, but they act as "black boxes." Regulatory bodies demand absolute mathematical transparency to prevent discriminatory lending and to provide clear reasons for loan rejections. 

The goal of this project is to build a credit scoring system that achieves the predictive accuracy of a modern gradient-boosting framework while maintaining the 100% mathematical explainability of a traditional banking scorecard.

## 💡 Our Solution
Instead of relying solely on black-box approximations, we engineered a custom **Double-Tree Econometric ML Model (Piecewise Linear Model)**. 

1. **The Segmenter:** A global decision tree segments borrowers into distinct macro-economic risk profiles based on historical bureau and application data.
2. **The Local Evaluator:** A dedicated Logistic Regression equation is dynamically routed and applied to that specific borrower's segment. 

To make this accessible to loan officers and risk executives, we developed and deployed a **Serverless Auditing Dashboard**. The dashboard allows officers to input a client ID, view the dynamic thresholding assessment, and see the exact mathematical impact (Client Value × Model Weight) of every variable on the final risk score.

## 🗺️ Methodology & Steps Taken
1. **Relational Data Aggregation:** Extracted and merged 7 relational tables (bureau data, previous applications, POS cash balances, credit cards) into a unified risk spine using a custom Two-Step Aggregation pipeline.
2. **Feature Engineering:** Derived high-value financial proxies (e.g., Credit Utilization Ratios, Payment Delays, Credit Hunger flags).
3. **Model Training & Ablation:** Trained three distinct architectures:
   - *Baseline:* Global Logistic Regression (White-box, low accuracy).
   - *Black-Box:* LightGBM + SHAP (High accuracy, poor global explainability).
   - *Our Solution:* Piecewise Model Tree (High accuracy, perfect local explainability).
4. **Dashboard Development:** Built a responsive, mobile-friendly Plotly Dash frontend to visualize the mathematical audit trails.
5. **Cloud Deployment:** Containerized the environment using Docker and Poetry, deploying the application as a serverless microservice on Google Cloud Run.

## 📖 Data & Feature Dictionary
Due to the vast amount of engineered features (combining internal bank data with external credit bureau records), the full data definitions and engineering logic have been documented separately.

* **Feature Engineering Documentation:** Please refer to [`data_feature_engineering.md`](./data_feature_engineering.md) for an in-depth breakdown of how historical aggregates were calculated.
* **Business Definitions:** The business-friendly descriptions used in the dashboard tooltips are sourced from `results/dashboard/feature_descriptions.txt`.

## ⚙️ System Requirements
To run this project locally or build the container, you will need:
* **Python:** `>=3.11, <3.13`
* **Dependency Manager:** [Poetry 2.x](https://python-poetry.org/)
* **Containerization:** Docker (if running via container)
* **Cloud:** Google Cloud CLI (if deploying to GCP)

## 🚀 How to Run the Project

### Option 1: Run Locally (Development Mode)
1. **Clone the repository and navigate to the root directory.**
2. **Install dependencies using Poetry:**
   ```bash
   poetry install

```

3. **Run the training pipeline (optional):**
*Note: Pre-trained models and sample dashboard data are already cached in `results/`.*
```bash
poetry run python scripts/pipline.py

```


4. **Launch the Dashboard:**
```bash
poetry run python results/dashboard/dashboard.py

```


*The dashboard will be available at `http://127.0.0.1:8050`.*

### Option 2: Run via Docker

1. **Build the Docker image:**
```bash
docker build -t credit-risk-dashboard .

```


2. **Run the container:**
```bash
docker run -p 8080:8080 credit-risk-dashboard

```


*The dashboard will be available at `http://localhost:8080`.*

### Option 3: Deploy to Google Cloud Run (Production)

Ensure you are authenticated with `gcloud` and have the required IAM permissions (`roles/storage.admin` and `roles/artifactregistry.writer`).

```bash
gcloud run deploy econometric-dashboard \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 4Gi

```

## 🛠️ Architecture & Tech Stack

* **Data Processing:** Pandas, NumPy, Category Encoders
* **Machine Learning:** Scikit-Learn, LightGBM, SHAP
* **Frontend UI:** Plotly Dash, Dash Bootstrap Components
* **Environment & Build:** Poetry 2.1, Docker, Gunicorn
* **Infrastructure:** Google Cloud Run (Serverless)
