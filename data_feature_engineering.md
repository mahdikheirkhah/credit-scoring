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



--- 

### data enginering