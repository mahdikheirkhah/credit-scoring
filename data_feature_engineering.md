### data realtion

# Exploratory Data Analysis: Understanding the Data Tables

This section provides an overview of the data tables available for the Home Credit Default Risk project. Before diving into numerical analysis and visualizations, it is crucial to understand the relationships between the tables, the type of information each contains, and how they can be connected to build a complete picture of each loan applicant.

## 1. Data Tables and Their Relationships

The dataset consists of one primary table (the main loan application) and six auxiliary tables that capture the applicant’s financial history from different sources. The following diagram illustrates the relationships:

```
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

- **`SK_ID_CURR`** is the unique identifier for each loan in the current sample (present in the main table and all auxiliary tables).
- **`SK_ID_PREV`** identifies a specific previous credit/application from Home Credit. It is used to link a previous application to its detailed monthly balances (POS, credit card) and its payment history.
- **`SK_BUREAU_ID`** identifies a specific credit record from the Credit Bureau and links the bureau table to its monthly balances in `bureau_balance`.

## 2. Description of Each Table

### 2.1 `application_{train|test}.csv` – The Main Application Table
- **Purpose**: Contains one row per loan application in the current sample. It holds static, application-time information about the client, the loan, and some external credit scores.
- **Key column**: `SK_ID_CURR` – unique loan ID.
- **Content groups**:
  - **Target variable** (train only): `TARGET` – 1 if the client had payment difficulties, 0 otherwise.
  - **Client demographics**: Gender, age (in days relative to application), family status, number of children, education, income, employment status, occupation, etc.
  - **Loan details**: Contract type (cash/revolving), credit amount, annuity, goods price.
  - **Client assets**: Car ownership, car age, realty ownership, housing type.
  - **Region information**: Normalized population, region ratings, address match flags.
  - **Application details**: Day and hour of application, provided documents flags (2–21), phone/email provision flags.
  - **External credit scores**: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` – normalized scores from external data sources.
  - **Building information**: Aggregated statistics (average, mode, median) about the building where the client lives (apartment size, area, elevators, entrances, construction year, etc.).
  - **Social circle defaults**: Counts of observations and defaults for 30/60 DPD in the client's social surroundings.
  - **Credit Bureau enquiries**: Number of enquiries to the Credit Bureau in the last hour, day, week, month, quarter, and year before the application.

### 2.2 `bureau.csv` – Credit Bureau Records
- **Purpose**: Contains all credit records from the Credit Bureau for the clients in the sample. Each client may have multiple previous credits reported by other financial institutions.
- **Relationship**: One `SK_ID_CURR` can be linked to multiple rows. Each row is identified by a unique `SK_BUREAU_ID`.
- **Content**:
  - **Credit status**: Active/closed status, days past due, maximal overdue amount.
  - **Timing**: Days before current application when the credit was granted, when it ended, when last updated.
  - **Financial details**: Credit amount, debt, limit, overdue amount, annuity.
  - **Credit characteristics**: Currency, type (car, cash, etc.), number of prolongations.

### 2.3 `bureau_balance.csv` – Monthly Balances of Credit Bureau Credits
- **Purpose**: Provides a month-by-month history of the credits reported in `bureau.csv`.
- **Relationship**: Linked to `bureau.csv` via `SK_BUREAU_ID`. For each credit, there is one row per month where history is available.
- **Content**:
  - `MONTHS_BALANCE`: Month relative to the current application date (e.g., -1 = most recent).
  - `STATUS`: Monthly status of the credit (C = closed, X = unknown, 0 = no DPD, 1 = DPD 1-30, 2 = DPD 31-60, …, 5 = DPD 120+ or sold/written off).

### 2.4 `POS_CASH_balance.csv` – Monthly Balances of Previous POS/Cash Loans
- **Purpose**: Contains monthly snapshots of previous point-of-sale (consumer) and cash loans that the applicant had with Home Credit.
- **Relationship**: Linked to the main table via `SK_ID_CURR` and to a specific previous credit via `SK_ID_PREV`. Multiple rows exist per previous credit (one per month).
- **Content**:
  - `MONTHS_BALANCE`: Month relative to application date.
  - `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`: Total and remaining instalments.
  - `NAME_CONTRACT_STATUS`: Contract status during that month (e.g., active, completed).
  - `SK_DPD`, `SK_DPD_DEF`: Days past due during the month (with and without tolerance for small amounts).

### 2.5 `credit_card_balance.csv` – Monthly Balances of Previous Credit Cards
- **Purpose**: Monthly snapshots of previous credit card accounts the applicant held with Home Credit.
- **Relationship**: Same structure as POS_CASH_balance – linked via `SK_ID_CURR` and `SK_ID_PREV`, with one row per month per credit card.
- **Content**:
  - `MONTHS_BALANCE`: Month relative to application date.
  - Balance and limit: `AMT_BALANCE`, `AMT_CREDIT_LIMIT_ACTUAL`.
  - Drawings: Amounts and counts of ATM, POS, and other drawings.
  - Payments: `AMT_PAYMENT_CURRENT`, `AMT_PAYMENT_TOTAL_CURRENT`.
  - Receivables and DPD: `AMT_RECEIVABLE_PRINCIPAL`, `SK_DPD`, `SK_DPD_DEF`.
  - `CNT_INSTALMENT_MATURE_CUM`: Number of paid instalments.
  - `NAME_CONTRACT_STATUS`: Contract status during the month.

### 2.6 `previous_application.csv` – Previous Home Credit Applications
- **Purpose**: Lists all prior loan applications the client made at Home Credit (whether approved, refused, or cancelled). One client may have multiple previous applications.
- **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`. Each row is one previous application.
- **Content**:
  - **Application details**: Contract type, application amount, final credit amount, down payment, goods price, annuity, purpose.
  - **Status and decision**: Contract status (approved, cancelled, refused), rejection reason, decision date.
  - **Timing**: Days relative to current application for decision, first drawing, first/last due, and expected termination.
  - **Product specifics**: Interest rates (normalized), payment type, whether insurance was requested, portfolio, product type, channel, seller details.
  - **Flags**: Whether it was the last application for the contract, last in a day, micro cash loan, etc.

### 2.7 `installments_payments.csv` – Payment History of Previous Home Credit Credits
- **Purpose**: Records the repayment history of previous credits disbursed by Home Credit. For each previous credit, it contains one row per payment made plus one row per missed payment.
- **Relationship**: Linked via `SK_ID_CURR` and `SK_ID_PREV`.
- **Content**:
  - **Instalment identification**: `NUM_INSTALMENT_VERSION` (version of payment calendar), `NUM_INSTALMENT_NUMBER` (instalment sequence).
  - **Timing**: When the instalment was supposed to be paid (`DAYS_INSTALMENT`) and when it was actually paid (`DAYS_ENTRY_PAYMENT`), both relative to the current application date.
  - **Amounts**: `AMT_INSTALMENT` (prescribed amount), `AMT_PAYMENT` (actual amount paid).

## 3. How the Tables Connect to Answer Business Questions

- **Client creditworthiness at application time** → `application_{train|test}.csv` (demographics, income, external scores).
- **Historical credit behavior outside Home Credit** → `bureau.csv` + `bureau_balance.csv` (past loans, payment statuses).
- **Past performance within Home Credit** → `previous_application.csv` (application outcomes) + `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` (detailed repayment behavior).

This structured view allows us to build a comprehensive feature set that captures both static client characteristics and dynamic payment behavior over time.


---

### data EDA

#### 1. `application_train.csv` Findings
Based on the initial exploratory data analysis of the main table (307,511 rows, 122 columns), we identified several critical issues:

1. **Extreme Outliers**: `AMT_INCOME_TOTAL` contains a massive outlier with a maximum value of 117,000,000 (117 Million), whereas the 75th percentile is only 202,500. This causes massive skew (391.56) and kurtosis (191,786.55), which would destroy mean/variance-based scalers like `StandardScaler`.
2. **Domain Anomalies**: `DAYS_EMPLOYED` has a maximum value of 365,243 (exactly 1000 years). This is a known systemic error code used when employment data is unavailable.
3. **High Missingness**: 67 columns contain missing values. In particular, building-related features (e.g., `COMMONAREA_MEDI`, `NONLIVINGAPARTMENTS_MODE`) are missing for nearly 70% of applicants.
4. **Redundant Information**: Building metrics are duplicated across three different aggregations (`_AVG`, `_MODE`, `_MEDI`), leading to high multicollinearity.
5. **Zero-Variance Features**: Some flags, such as `FLAG_MOBIL`, have only one unique value (1) across the entire dataset, providing zero predictive power.

---


#### 2. `bureau.csv` & `bureau_balance.csv` Findings
Based on the exploratory data analysis of the credit bureau tables (1,716,428 records), we identified several key characteristics common to financial history data:

1. **Extreme Missingness in Specific Fields**: Features like `AMT_ANNUITY` (71.47% missing) and `AMT_CREDIT_MAX_OVERDUE` (65.51% missing) are heavily unpopulated. Other fields like `DAYS_ENDDATE_FACT` (36.9%) and `AMT_CREDIT_SUM_LIMIT` (34.4%) also have significant gaps.
2. **Massive Outliers**: Fields such as `AMT_CREDIT_MAX_OVERDUE` contain extreme outliers (maximum value of 115,987,200), and `CREDIT_DAY_OVERDUE` has values up to 2,792 days.
3. **Highly Asymmetric Distributions**: As expected in credit risk, the vast majority of clients have zero overdue days and zero bad debt, creating massive positive skew and kurtosis in default-related columns.


---

#### 3. `POS_CASH_balance.csv` Findings

Unlike the external bureau data, this table represents Home Credit's internal ledger, leading to distinct data characteristics:

1. **Negligible Missingness**: Because it is an internal system tracking monthly point-of-sale and cash loans, there is virtually no missing data.
2. **Behavioral Outliers**: Columns like `SK_DPD` (Days Past Due) and `SK_DPD_DEF` (Days Past Due with Tolerance) contain extreme outliers (e.g., thousands of days past due). These are not system errors; they represent factual, severe loan abandonments.
3. **Term-Length Restructuring (`CNT_INSTALMENT`)**: Sudden spikes or extreme values in the expected number of installments often indicate that a struggling client had their loan term extended (restructured) to lower monthly payments.

---

--- 

### data enginering

#### 1. `application_train.csv` Pipeline Solutions
To resolve the EDA findings while strictly adhering to `Contributing.md` rules (especially zero data leakage), the following preprocessing architecture was implemented in `scripts/preprocess.py`:

1. **Robust Scaling (`RobustScaler`)**: To handle the $117M extreme income outlier without manually manipulating the raw data, we swapped `StandardScaler` for `RobustScaler`. It relies on the Median and Interquartile Range (IQR), natively ignoring massive outliers and protecting the model.
2. **The Missingness Signal (`add_indicator=True`)**: Because 70% of building data is missing, the *absence* of data is highly predictive (e.g., clients who don't provide building data might be higher risk). We used Scikit-Learn's `SimpleImputer(add_indicator=True)` to impute missing values with the median while natively generating a binary `_MISSING` flag to preserve the signal.
3. **Static Domain Fixes**: Hardcoded a replacement of the 365,243 `DAYS_EMPLOYED` anomaly with `NaN` inside the `load_data` method, routing it to our intelligent imputer.
4. **Stateful Feature Filtering**: Created a custom `filter_features(X, fit=bool)` method to drop highly correlated redundant columns (`_AVG`, `_MODE`) and zero-variance columns (`FLAG_MOBIL`). Crucially, this method *memorizes* the dropped columns during training (`fit=True`) and perfectly applies the same drops to the test set (`fit=False`), completely preventing pipeline schema mismatches and data leakage.

---

#### 2. `bureau.csv` Pipeline Solutions
Our architecture elegantly handles these issues without requiring messy, manual pandas code:

1. **Handling Missing Values during Aggregation**: 
   - When we aggregate the bureau records down to the client level in `scripts/feature_engineering.py` using functions like `.groupby().agg(['mean', 'max', 'sum'])`, pandas **natively ignores NaNs**. 
   - If a client has 5 historical loans but only 1 contains `AMT_ANNUITY` data, the mean is safely calculated from that 1 value.
   - If all historical records are NaN, the resulting aggregated feature will also be `NaN`. These remaining NaNs are passed seamlessly into our main `BaselinePreprocessor`, where `SimpleImputer(add_indicator=True)` takes over. It fills the `NaN` with the median, but creates a binary `_MISSING` flag. This preserves the "missingness signal" (i.e., the fact that the credit bureau has no data on their past annuity is itself a highly predictive risk feature).
2. **Handling Massive Outliers**:
   - We do *not* need to manually cap (Winsorize) these 115 million outliers. Because our preprocessing pipeline feeds all aggregated features through the `RobustScaler`, the scaler will use the Interquartile Range (25th to 75th percentile) to scale the data, completely ignoring the extreme values.
3. **Handling Asymmetric Distributions**:
   - Credit data is inherently skewed. Our primary predictive engine, **LightGBM (Gradient Boosted Trees)**, is fundamentally invariant to monotonic transformations and handles non-normal, heavily skewed distributions natively by splitting on thresholds (e.g., `OVERDUE_MAX > 30`) rather than relying on variance equations. For our linear baselines (Logistic Regression), the `RobustScaler` provides sufficient stabilization.


---

#### 3. `bureau.csv` & `bureau_balance.csv` Behavioral Feature Extraction

Beyond pipeline handling, we explicitly engineered domain-specific behavioral flags before passing the data to the preprocessor:

1. **The Revolving Credit Signal**: We identified that `DAYS_CREDIT_ENDDATE` values exceeding 10,000 days (27+ years) are system placeholders for Open-Ended/Credit Card debt. We extracted this into a binary `IS_OPEN_ENDED_CREDIT` flag to preserve the signal, then replaced the extreme values with `NaN` to protect the mean duration calculations.
2. **Recent Credit Hunger**: Engineered a `RECENT_LOAN_FLAG` identifying loans applied for in the last 180 days, capturing sudden desperation for liquidity.
3. **Active Debt Burden**: Raw debt-to-credit ratios are skewed by closed loans (which have 0 debt). We isolated only `Active` loans to calculate a highly accurate `BUREAU_ACTIVE_DEBT_RATIO`.
4. **Time-Series Flattening**: The 27 million `bureau_balance` rows were flattened by converting categorical statuses into mathematical severity flags (`STATUS_CLOSED`, `STATUS_SEVERE_DPD`), then aggregating them by loan ID before merging them into the main bureau table.


--- 

#### 4. `POS_CASH_balance.csv` Pipeline Solutions

To capture the severe delinquencies without manually capping the true financial behavior:

1. **Delinquency Thresholding**: Instead of feeding raw days-past-due into the model, we engineered boolean behavioral flags: `POS_IS_LATE` (`SK_DPD_DEF > 0`) and `POS_IS_SEVERE_LATE` (`SK_DPD_DEF > 30`).
2. **Frequency Aggregation**: By aggregating these flags using `sum` and `mean` at the client level, we converted chaotic daily time-series data into measurable frequencies of financial distress (e.g., "This client was severely late for 14 total months").
3. **Magnitude Preservation**: By passing the raw maximum values through our `RobustScaler`, we allowed the tree-based models to see the true severity of the worst defaults without destroying the linear model's variance.
