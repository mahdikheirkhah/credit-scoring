# Home Credit Default Risk: Data Architecture & Engineering

## Table of Contents

1. [Part 1: Data Architecture & Relationships](#part-1-data-architecture--relationships)
2. [Part 2: Exploratory Data Analysis (EDA) Findings](#part-2-exploratory-data-analysis-eda-findings)
3. [Part 3: Data Engineering & Pipeline Solutions](#part-3-data-engineering--pipeline-solutions)

---

## Part 1: Data Architecture & Relationships

This section provides an overview of the data tables available for the project. Understanding the relationships between the tables and the type of information each contains is crucial for building a complete picture of each loan applicant.

### 1.1 Data Tables and Their Relationships

The dataset consists of one primary table (the main loan application) and six auxiliary tables that capture the applicant’s financial history from different sources.

```text
 application_{train|test}.csv  (main table)
         │  PK: SK_ID_CURR
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
   bureau.csv                          previous_application.csv
   (credit bureau records)             (previous Home Credit applications)
   FK: SK_ID_CURR                      FK: SK_ID_CURR, SK_ID_PREV
         │                                     │
         │ (SK_BUREAU_ID)                      │ (SK_ID_PREV)
         ▼                                     │
   bureau_balance.csv                          ├─── POS_CASH_balance.csv
   (monthly snapshots of bureau                │    (monthly balances of POS/cash loans)
    credits)                                   │    FK: SK_ID_CURR, SK_ID_PREV
                                               │
                                               ├─── credit_card_balance.csv
                                               │    (monthly balances of credit cards)
                                               │    FK: SK_ID_CURR, SK_ID_PREV
                                               │
                                               └─── installments_payments.csv
                                                    (payment history of previous credits)
                                                    FK: SK_ID_CURR, SK_ID_PREV

```

* **`SK_ID_CURR`**: The unique identifier for each loan in the current sample (present in the main table and all auxiliary tables).
* **`SK_ID_PREV`**: Identifies a specific previous credit/application from Home Credit. Used to link a previous application to its detailed monthly balances (POS, credit card) and payment history.
* **`SK_BUREAU_ID`**: Identifies a specific credit record from the Credit Bureau and links the bureau table to its monthly balances in `bureau_balance`.

### 1.2 Description of Each Table

<a name="def-app-train"></a>
### `application_{train|test}.csv` – The Main Application Table

* **Purpose**: Contains one row per loan application in the current sample. It holds static, application-time information about the client, the loan, and external credit scores.
* **Key column**: `SK_ID_CURR` – unique loan ID.
* **Target variable (train only)**: `TARGET` – 1 if the client had payment difficulties, 0 otherwise.
* **Client demographics**: Gender, age (in days relative to application), family status, number of children, education, income, employment status, occupation.
* **Loan details**: Contract type (cash/revolving), credit amount, annuity, goods price.
* **Client assets**: Car ownership, car age, realty ownership, housing type.
* **Region information**: Normalized population, region ratings, address match flags.
* **Application details**: Day and hour of application, provided documents flags (2–21), phone/email provision flags.
* **External credit scores**: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` – normalized scores from external data sources.
* **Building information**: Aggregated statistics (average, mode, median) about the building where the client lives.
* **Social circle defaults**: Counts of observations and defaults for 30/60 DPD in the client's social surroundings.
* **Credit Bureau enquiries**: Number of enquiries to the Credit Bureau in the last hour, day, week, month, quarter, and year before the application.

<a name="def-bureau"></a>
### `bureau.csv` – Credit Bureau Records

* **Purpose**: Contains all credit records from the Credit Bureau for the clients in the sample. Each client may have multiple previous credits reported by other financial institutions.
* **Relationship**: Linked via `SK_ID_CURR`. Identified by a unique `SK_BUREAU_ID`.
* **Credit status**: Active/closed status, days past due, maximal overdue amount.
* **Timing**: Days before current application when the credit was granted, when it ended, when last updated.
* **Financial details**: Credit amount, debt, limit, overdue amount, annuity.
* **Credit characteristics**: Currency, type (car, cash, etc.), number of prolongations.

<a name="def-bureau-balance"></a>
### `bureau_balance.csv` – Monthly Balances of Credit Bureau Credits

* **Purpose**: Provides a month-by-month history of the credits reported in `bureau.csv`.
* **Relationship**: Linked to `bureau.csv` via `SK_BUREAU_ID`.
* **MONTHS_BALANCE**: Month relative to the current application date (e.g., -1 = most recent).
* **STATUS**: Monthly status of the credit (C = closed, X = unknown, 0 = no DPD, 1 = DPD 1-30, 2 = DPD 31-60, …, 5 = DPD 120+ or sold/written off).

<a name="def-pos-cash"></a>
### `POS_CASH_balance.csv` – Monthly Balances of Previous POS/Cash Loans

* **Purpose**: Contains monthly snapshots of previous point-of-sale (consumer) and cash loans that the applicant had with Home Credit.
* **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`.
* **MONTHS_BALANCE**: Month relative to application date.
* **CNT_INSTALMENT / CNT_INSTALMENT_FUTURE**: Total and remaining instalments.
* **NAME_CONTRACT_STATUS**: Contract status during that month (e.g., active, completed).
* **SK_DPD / SK_DPD_DEF**: Days past due during the month (with and without tolerance for small amounts).

<a name="def-credit-card"></a>
### `credit_card_balance.csv` – Monthly Balances of Previous Credit Cards

* **Purpose**: Monthly snapshots of previous credit card accounts the applicant held with Home Credit.
* **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`.
* **MONTHS_BALANCE**: Month relative to application date.
* **Balance and limit**: `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL`.
* **Drawings**: Amounts and counts of ATM, POS, and other drawings.
* **Payments**: `AMT_PAYMENT_CURRENT`, `AMT_PAYMENT_TOTAL_CURRENT`.
* **Receivables and DPD**: `AMT_RECEIVABLE_PRINCIPAL`, `SK_DPD`, `SK_DPD_DEF`.
* **CNT_INSTALMENT_MATURE_CUM**: Number of paid instalments.
* **NAME_CONTRACT_STATUS**: Contract status during the month.

<a name="def-prev-app"></a>
### `previous_application.csv` – Previous Home Credit Applications

* **Purpose**: Lists all prior loan applications the client made at Home Credit (whether approved, refused, or cancelled).
* **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`.
* **Application details**: Contract type, application amount, final credit amount, down payment, goods price, annuity, purpose.
* **Status and decision**: Contract status (approved, cancelled, refused), rejection reason, decision date.
* **Timing**: Days relative to current application for decision, first drawing, first/last due, and expected termination.
* **Product specifics**: Interest rates (normalized), payment type, whether insurance was requested, portfolio, product type, channel, seller details.
* **Flags**: Whether it was the last application for the contract, last in a day, micro cash loan, etc.

<a name="def-installments"></a>
### `installments_payments.csv` – Payment History of Previous Home Credit Credits

* **Purpose**: Records the repayment history of previous credits disbursed by Home Credit.
* **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`.
* **Instalment identification**: `NUM_INSTALMENT_VERSION` (version of payment calendar), `NUM_INSTALMENT_NUMBER` (instalment sequence).
* **Timing**: When the instalment was supposed to be paid (`DAYS_INSTALMENT`) and when it was actually paid (`DAYS_ENTRY_PAYMENT`), both relative to the current application date.
* **Amounts**: `AMT_INSTALMENT` (prescribed amount), `AMT_PAYMENT` (actual amount paid).

### 1.3 How the Tables Connect to Answer Business Questions

* **Client creditworthiness at application time**: Analyzed via [`application_{train|test}.csv`](#def-app-train) (demographics, income, external scores).
* **Historical credit behavior outside Home Credit**: Analyzed via [`bureau.csv`](#def-bureau) + [`bureau_balance.csv`](#def-bureau-balance) (past loans, payment statuses).
* **Past performance within Home Credit**: Analyzed via [`previous_application.csv`](#def-prev-app) (application outcomes) + [`POS_CASH_balance.csv`](#def-pos-cash), [`credit_card_balance.csv`](#def-credit-card), [`installments_payments.csv`](#def-installments) (detailed repayment behavior).

---

## Part 2: Exploratory Data Analysis (EDA) Findings

### 2.1 Findings: [`application_train.csv`](#def-app-train)

1. **Extreme Outliers**: `AMT_INCOME_TOTAL` contains a massive outlier with a maximum value of 117,000,000 (117 Million), whereas the 75th percentile is only 202,500. This causes massive skew (391.56) and kurtosis (191,786.55).
2. **Domain Anomalies**: `DAYS_EMPLOYED` has a maximum value of 365,243 (exactly 1000 years). This is a known systemic error code used when employment data is unavailable.
3. **High Missingness**: 67 columns contain missing values. Building-related features (e.g., `COMMONAREA_MEDI`, `NONLIVINGAPARTMENTS_MODE`) are missing for nearly 70% of applicants.
4. **Redundant Information**: Building metrics are duplicated across three different aggregations (`_AVG`, `_MODE`, `_MEDI`), leading to high multicollinearity.
5. **Zero-Variance Features**: Flags such as `FLAG_MOBIL` have only one unique value (1) across the entire dataset, providing zero predictive power.

### 2.2 Findings: [`bureau.csv`](#def-bureau) & [`bureau_balance.csv`](#def-bureau-balance)

1. **Extreme Missingness in Specific Fields**: Features like `AMT_ANNUITY` (71.47% missing) and `AMT_CREDIT_MAX_OVERDUE` (65.51% missing) are heavily unpopulated. Fields like `DAYS_ENDDATE_FACT` (36.9%) and `AMT_CREDIT_SUM_LIMIT` (34.4%) also have significant gaps.
2. **Massive Outliers**: Fields such as `AMT_CREDIT_MAX_OVERDUE` contain extreme outliers (maximum value of 115,987,200), and `CREDIT_DAY_OVERDUE` has values up to 2,792 days.
3. **Highly Asymmetric Distributions**: The vast majority of clients have zero overdue days and zero bad debt, creating massive positive skew and kurtosis in default-related columns.

### 2.3 Findings: [`POS_CASH_balance.csv`](#def-pos-cash)

1. **Negligible Missingness**: Because it is an internal system tracking monthly point-of-sale and cash loans, there is virtually no missing data.
2. **Behavioral Outliers**: Columns like `SK_DPD` (Days Past Due) and `SK_DPD_DEF` (Days Past Due with Tolerance) contain extreme outliers (e.g., thousands of days past due). These are factual, severe loan abandonments, not system errors.
3. **Term-Length Restructuring (`CNT_INSTALMENT`)**: Sudden spikes or extreme values in the expected number of installments indicate struggling clients had their loan term extended (restructured) to lower monthly payments.

### 2.4 Findings: [`credit_card_balance.csv`](#def-credit-card)

1. **The "Inactive Account" Missing Data Pattern**: A block of variables related to drawings and payments is missing exactly ~19.5% of their values. Domain analysis reveals this represents months where the credit card was entirely inactive.
2. **Behavioral Outliers & Cash Desperation**: The data contains massive outliers in `AMT_DRAWINGS_ATM_CURRENT`, signaling high-risk cash desperation.

### 2.5 Findings: [`installments_payments.csv`](#def-installments)

1. **The "Missed Payment" Anomaly**: EDA revealed exactly 0.02% missing values in `AMT_PAYMENT` and `DAYS_ENTRY_PAYMENT`. These are not data collection errors, but actual abandoned/missed installments.
2. **Extreme Outliers as Valid Signals**: The `PAYMENT_DELAY` feature exhibited massive bidirectional outliers (e.g., -3,189 days early and +2,884 days late). These are valid structural signals representing Lump Sum Prepayments vs. Debt Collections.
3. **Payment Outliers**: Massive outliers exist in `AMT_PAYMENT` (up to $3.7M).

### 2.6 Findings: [`previous_application.csv`](#def-prev-app)

1. **The 40% Missing Block**: Temporal columns indicating fund disbursement and termination were missing for exactly 40.30% of records. This correlates directly to applications that were Refused or Canceled.
2. **The 365,243 "To Be Determined" Flag**: The 1,000-year anomaly returned in the temporal columns, representing active loans that have not yet reached their termination state.
3. **Backend Payload Outliers**: `SELLERPLACE_AREA` exhibited extreme outliers (up to 4,000,000). Domain review indicates these are likely merchant API routing errors where postal codes or IDs populated the integer area field.

---

## Part 3: Data Engineering & Pipeline Solutions

### 3.1 Pipeline Solutions: [`application_train.csv`](#def-app-train)

1. **Robust Scaling (`RobustScaler`)**: To handle extreme income outliers without manual manipulation, `StandardScaler` was swapped for `RobustScaler`. It relies on the Median and Interquartile Range (IQR), natively ignoring massive outliers.
2. **The Missingness Signal (`add_indicator=True`)**: Because 70% of building data is missing, the absence of data is highly predictive. Scikit-Learn's `SimpleImputer(add_indicator=True)` imputes missing values with the median while natively generating a binary `_MISSING` flag.
3. **Static Domain Fixes**: Hardcoded a replacement of the 365,243 `DAYS_EMPLOYED` anomaly with `NaN` inside the `load_data` method to route it to the imputer.
4. **Stateful Feature Filtering**: Created a custom `filter_features(X, fit=bool)` method to drop highly correlated redundant columns (`_AVG`, `_MODE`) and zero-variance columns (`FLAG_MOBIL`). This method memorizes dropped columns during training to perfectly apply the same schema to test data, preventing leakage.

### 3.2 Pipeline Solutions: [`bureau.csv`](#def-bureau) & [`bureau_balance.csv`](#def-bureau-balance)

1. **Handling Missing Values during Aggregation**: Pandas natively ignores NaNs during `.groupby().agg()`. Resulting all-NaN client rows pass seamlessly into the `BaselinePreprocessor`, where `SimpleImputer(add_indicator=True)` creates predictive `_MISSING` flags.
2. **Handling Massive Outliers**: `RobustScaler` dynamically handles extreme values (like 115 million outliers) via IQR scaling, eliminating the need for manual capping/Winsorizing.
3. **Handling Asymmetric Distributions**: LightGBM fundamentally handles non-normal, heavily skewed distributions natively by splitting on thresholds. The linear baselines rely on `RobustScaler` stabilization.
4. **The Revolving Credit Signal**: Extracted `DAYS_CREDIT_ENDDATE` values exceeding 10,000 days into a binary `IS_OPEN_ENDED_CREDIT` flag to preserve the credit card signal, replacing the raw extremes with `NaN`.
5. **Recent Credit Hunger**: Engineered a `RECENT_LOAN_FLAG` identifying loans applied for in the last 180 days.
6. **Active Debt Burden**: Isolated only `Active` loans to calculate an accurate `BUREAU_ACTIVE_DEBT_RATIO`, preventing closed loans from skewing the math.
7. **Time-Series Flattening**: Converted categorical statuses in the 27M `bureau_balance` rows into mathematical severity flags (`STATUS_CLOSED`, `STATUS_SEVERE_DPD`), aggregating them by loan ID before merging into the main bureau table.

### 3.3 Pipeline Solutions: [`POS_CASH_balance.csv`](#def-pos-cash)

1. **Delinquency Thresholding**: Engineered boolean behavioral flags: `POS_IS_LATE` (`SK_DPD_DEF > 0`) and `POS_IS_SEVERE_LATE` (`SK_DPD_DEF > 30`) instead of feeding raw days-past-due into the model.
2. **Frequency Aggregation**: Aggregated these flags using `sum` and `mean` at the client level to convert chaotic daily time-series data into measurable frequencies of financial distress.
3. **Magnitude Preservation**: Passed the raw maximum values through `RobustScaler` to allow tree-based models to see the true severity of the worst defaults without destroying linear variance.

### 3.4 Pipeline Solutions: [`credit_card_balance.csv`](#def-credit-card)

1. **Handling "Inactive" Missing Data**: Explicitly created a `_WAS_MISSING` binary flag to count "inactive months" as a predictive feature, filling the raw columns with `0.0` before aggregation to reflect true zero-activity months.
2. **Guardrailed Ratios**: Engineered `CC_ATM_DRAWING_RATIO` (ATM Drawings / Credit Limit) and `CC_UTILIZATION` ratios. Replaced `$0` limits with `NaN` to prevent Infinity errors, and hard-clipped ratios at `1.5` for model stability.
3. **Two-Step Aggregation Architecture**: To prevent "data blurring" across multiple cards, data was aggregated first to the Card level (`SK_ID_PREV`) to calculate per-card maximums, and then to the Client level (`SK_ID_CURR`) to capture features like "Highest Utilization Across Any Single Card."

### 3.5 Pipeline Solutions: [`installments_payments.csv`](#def-installments)

1. **The "Missed Payment" Extraction**: Engineered an explicit `IS_MISSED_PAYMENT` flag and filled missing `AMT_PAYMENT` rows with `0.0` to capture severe default behavior without relying on median imputation.
2. **Validating Extreme Outliers**: Processed the massive bidirectional `PAYMENT_DELAY` outliers as valid structural signals (Lump Sum Prepayments vs. Debt Collections) via `RobustScaler`.
3. **Payment Fraction Engineering**: Neutralized massive `AMT_PAYMENT` outliers by engineering a `PAYMENT_FRACTION` (`AMT_PAYMENT / AMT_INSTALMENT`) to detect Underpayment, Exact Payment, or Prepayment.
4. **Two-Step Aggregation**: Calculated worst-case delays and minimum payment fractions at the individual loan level (`SK_ID_PREV`), then rolled those extreme metrics up to the client level (`SK_ID_CURR`).

### 3.6 Pipeline Solutions: [`previous_application.csv`](#def-prev-app)

1. **State Machine Logic for Missing Blocks**: Engineered explicit approval, refusal, and cancellation ratios (`PREV_IS_REFUSED_MEAN`) to mathematically capture the structural state of the 40.30% missing block.
2. **Neutralizing the 365,243 Flag**: Programmatically replaced the 1,000-year anomaly in temporal columns with `NaN` before applying mean aggregations to prevent data distortion.
3. **The "Credit Ask" Ratio**: Engineered an `APPLICATION_CREDIT_RATIO` (Amount Applied / Amount Granted). Ratios greater than `1.0` indicate financial overextension.
4. **Pre-Aggregation Missingness Tracking**: Explicitly engineered pre-aggregation boolean flags (`HAS_DOWN_PAYMENT`, `HAS_DISBURSEMENT_INFO`) and aggregated their `mean()` to capture behaviors like "provided a down payment exactly 20% of the time," which would otherwise be erased by NaN-ignoring aggregations. Isolated system flags like `SELLERPLACE_AREA = -1` into an `AREA_WAS_NEGATIVE` indicator.
