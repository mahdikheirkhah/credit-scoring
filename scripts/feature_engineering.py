import pandas as pd
from abc import ABC, abstractmethod
from loguru import logger


# ---------------------------------------------------------
# 1. THE INTERFACE (Contract)
# ---------------------------------------------------------
class FeatureAggregator(ABC):
    """
    Abstract Base Class for all relational data aggregators.
    Forces all child classes to implement the `aggregate` method.
    """

    @abstractmethod
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a raw relational DataFrame and returns flattened features."""
        pass


# ---------------------------------------------------------
# 2. THE CONCRETE IMPLEMENTATIONS
# ---------------------------------------------------------
class BureauAggregator(FeatureAggregator):
    """Handles specific business logic for external credit bureau data."""

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating bureau.csv (External Credit History)...")

            df_working = df.copy()

            # --- BEHAVIORAL FEATURE 1: Recent Credit Hunger ---
            if "DAYS_CREDIT" in df_working.columns:
                df_working["RECENT_LOAN_FLAG"] = (
                    df_working["DAYS_CREDIT"] > -180
                ).astype(int)

            # --- BEHAVIORAL FEATURE 2: Overdue History ---
            if "AMT_CREDIT_MAX_OVERDUE" in df_working.columns:
                df_working["HAS_BEEN_OVERDUE"] = (
                    df_working["AMT_CREDIT_MAX_OVERDUE"] > 0
                ).astype(int)

            # System Placeholder Outlier Detection
            if "DAYS_CREDIT_ENDDATE" in df_working.columns:
                # Flag revolving/open-ended credit (System anomalies like 31,199 days)
                df_working["IS_OPEN_ENDED_CREDIT"] = (
                    df_working["DAYS_CREDIT_ENDDATE"] > 10000
                ).astype(int)
                # Replace extreme outlier with NaN so it doesn't wreck downstream metrics
                df_working.loc[
                    df_working["DAYS_CREDIT_ENDDATE"] > 10000, "DAYS_CREDIT_ENDDATE"
                ] = float("nan")

            aggregations = {
                "SK_ID_BUREAU": ["count"],
                "DAYS_CREDIT": ["mean", "min", "max"],
                "AMT_CREDIT_SUM": ["sum", "mean"],
                "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
            }

            if "DAYS_CREDIT_ENDDATE" in df_working.columns:
                aggregations["DAYS_CREDIT_ENDDATE"] = ["mean", "max"]
            if "IS_OPEN_ENDED_CREDIT" in df_working.columns:
                aggregations["IS_OPEN_ENDED_CREDIT"] = ["sum", "max"]
            if "RECENT_LOAN_FLAG" in df_working.columns:
                aggregations["RECENT_LOAN_FLAG"] = ["sum"]
            if "HAS_BEEN_OVERDUE" in df_working.columns:
                aggregations["HAS_BEEN_OVERDUE"] = ["max"]

            # If bureau balance features have been merged in, roll them up to SK_ID_CURR
            bal_cols = [c for c in df_working.columns if c.startswith("BUREAU_BAL_")]
            for col in bal_cols:
                aggregations[col] = ["mean", "sum"]

            agg_df = df_working.groupby("SK_ID_CURR").agg(aggregations)
            agg_df.columns = pd.Index(
                [f"BUREAU_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()]
            )

            if (
                "BUREAU_AMT_CREDIT_SUM_DEBT_SUM" in agg_df
                and "BUREAU_AMT_CREDIT_SUM_SUM" in agg_df
            ):
                agg_df["BUREAU_DEBT_CREDIT_RATIO"] = agg_df[
                    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM"
                ] / agg_df["BUREAU_AMT_CREDIT_SUM_SUM"].replace(0, float("nan"))

            # --- BEHAVIORAL FEATURE 3: Active Debt Ratio ---
            if "CREDIT_ACTIVE" in df_working.columns:
                active_df = df_working[df_working["CREDIT_ACTIVE"] == "Active"]
                active_agg = active_df.groupby("SK_ID_CURR").agg(
                    {"AMT_CREDIT_SUM": ["sum"], "AMT_CREDIT_SUM_DEBT": ["sum"]}
                )
                active_agg.columns = [
                    "BUREAU_ACTIVE_CREDIT_SUM",
                    "BUREAU_ACTIVE_DEBT_SUM",
                ]

                active_agg["BUREAU_ACTIVE_DEBT_RATIO"] = active_agg[
                    "BUREAU_ACTIVE_DEBT_SUM"
                ] / active_agg["BUREAU_ACTIVE_CREDIT_SUM"].replace(0, float("nan"))

                agg_df = agg_df.join(
                    active_agg[["BUREAU_ACTIVE_DEBT_RATIO"]], how="left"
                )

            logger.info(
                f"Successfully engineered Bureau proxies. Shape: {agg_df.shape}"
            )
            return agg_df.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate bureau data: {e}")
            raise


class PreviousApplicationAggregator(FeatureAggregator):
    """Handles specific business logic for internal previous applications."""

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating previous_application.csv (Internal History)...")

            # We use .copy() to avoid SettingWithCopyWarning
            df_working = df.copy()
            df_working["IS_APPROVED"] = (
                df_working["NAME_CONTRACT_STATUS"] == "Approved"
            ).astype(int)
            df_working["IS_REFUSED"] = (
                df_working["NAME_CONTRACT_STATUS"] == "Refused"
            ).astype(int)

            aggregations = {
                "SK_ID_PREV": ["count"],
                "AMT_APPLICATION": ["mean", "max"],
                "IS_APPROVED": ["mean", "sum"],
                "IS_REFUSED": ["mean", "sum"],
            }
            agg_df = df_working.groupby("SK_ID_CURR").agg(aggregations)
            agg_df.columns = pd.Index(
                [f"PREV_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()]
            )

            logger.info(
                f"Successfully engineered Previous App proxies. Shape: {agg_df.shape}"
            )
            return agg_df.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate previous application data: {e}")
            raise


class POSCashAggregator(FeatureAggregator):
    """Handles the 1:N:M relationship for internal POS and Cash loan monthly balances."""

    def aggregate(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(
                "Aggregating POS_CASH_balance.csv (Internal Monthly History)..."
            )

            # Create a working copy
            df_pos = pos_df.copy()

            # --- BEHAVIORAL FEATURE 1: Severe Delinquency ---
            # SK_DPD_DEF ignores small fees, so > 0 means a real, actionable missed payment
            df_pos["POS_IS_LATE"] = (df_pos["SK_DPD_DEF"] > 0).astype(int)
            # Flag severe defaults (more than 30 days late)
            df_pos["POS_IS_SEVERE_LATE"] = (df_pos["SK_DPD_DEF"] > 30).astype(int)

            # --- BEHAVIORAL FEATURE 2: Account Status Tracking ---
            df_pos["POS_IS_ACTIVE"] = (
                df_pos["NAME_CONTRACT_STATUS"] == "Active"
            ).astype(int)
            df_pos["POS_IS_COMPLETED"] = (
                df_pos["NAME_CONTRACT_STATUS"] == "Completed"
            ).astype(int)

            # --- AGGREGATION: Flatten the history to the Client Level ---
            aggregations = {
                "MONTHS_BALANCE": [
                    "max",
                    "min",
                    "size",
                ],  # Recency and length of history
                "SK_DPD_DEF": ["max", "mean"],  # The absolute worst they've ever been
                "POS_IS_LATE": [
                    "sum",
                    "mean",
                ],  # Total months late, and % of months late
                "POS_IS_SEVERE_LATE": ["sum"],  # Count of severely delinquent months
                "CNT_INSTALMENT": ["mean", "max"],  # Average and max term length
                "CNT_INSTALMENT_FUTURE": ["mean"],
                "POS_IS_COMPLETED": ["sum"],  # Count of successfully completed months
                "POS_IS_ACTIVE": ["sum"],
            }

            agg_df = df_pos.groupby("SK_ID_CURR").agg(aggregations)

            # Flatten MultiIndex columns
            agg_df.columns = pd.Index(
                [f"POS_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()]
            )

            logger.info(
                f"Successfully engineered POS Cash proxies. Shape: {agg_df.shape}"
            )
            return agg_df.reset_index()

        except Exception as e:
            logger.error(f"Failed to aggregate POS Cash data: {e}")
            raise


class CreditCardAggregator(FeatureAggregator):
    """Handles the 1:N:M relationship for internal Credit Card monthly balances using Two-Step Aggregation."""

    def aggregate(self, cc_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating credit_card_balance.csv (Two-Step Aggregation)...")
            df_cc = cc_df.copy()

            # --- 1. EXPLICIT MISSINGNESS & DOMAIN FILLS ---
            fill_0_cols = [
                "AMT_DRAWINGS_ATM_CURRENT",
                "AMT_DRAWINGS_OTHER_CURRENT",
                "AMT_DRAWINGS_POS_CURRENT",
                "CNT_DRAWINGS_ATM_CURRENT",
                "CNT_DRAWINGS_OTHER_CURRENT",
                "CNT_DRAWINGS_POS_CURRENT",
                "AMT_PAYMENT_CURRENT",
            ]
            for col in fill_0_cols:
                if col in df_cc.columns:
                    # Capture the "Inactive Month" signal before filling
                    df_cc[f"{col}_WAS_MISSING"] = df_cc[col].isna().astype(int)
                    df_cc[col] = df_cc[col].fillna(0)

            # --- 2. SAFE RATIO ENGINEERING ---
            # Replace 0 limit with NaN to prevent divide-by-zero Infinity
            df_cc["SAFE_LIMIT"] = df_cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(
                0, float("nan")
            )
            df_cc["CC_UTILIZATION"] = df_cc["AMT_BALANCE"] / df_cc["SAFE_LIMIT"]
            df_cc["CC_ATM_DRAWING_RATIO"] = (
                df_cc["AMT_DRAWINGS_ATM_CURRENT"] / df_cc["SAFE_LIMIT"]
            )

            # Guardrail: Cap astronomical ratios (e.g., caused by $1 limits + fees)
            df_cc["CC_UTILIZATION"] = df_cc["CC_UTILIZATION"].clip(upper=1.5)
            df_cc["CC_ATM_DRAWING_RATIO"] = df_cc["CC_ATM_DRAWING_RATIO"].clip(
                upper=1.5
            )

            # Delinquency Flags
            df_cc["CC_IS_LATE"] = (df_cc["SK_DPD_DEF"] > 0).astype(int)
            df_cc["CC_IS_SEVERE_LATE"] = (df_cc["SK_DPD_DEF"] > 30).astype(int)

            # ==========================================================
            # STEP 1: AGGREGATE TO CARD LEVEL (SK_ID_PREV)
            # ==========================================================
            card_level_aggs = {
                "MONTHS_BALANCE": ["size"],  # Lifespan of this specific card
                "CC_UTILIZATION": ["max", "mean"],  # Did THIS card max out?
                "CC_ATM_DRAWING_RATIO": ["max"],  # Peak desperation on this card
                "CC_IS_LATE": ["sum"],
                "CC_IS_SEVERE_LATE": ["sum"],
                "AMT_DRAWINGS_ATM_CURRENT_WAS_MISSING": [
                    "sum"
                ],  # Months this card was inactive
            }
            # We group by both to keep CURR available for Step 2
            card_agg = df_cc.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(card_level_aggs)
            card_agg.columns = pd.Index(
                [f"CARD_{e[0]}_{e[1].upper()}" for e in card_agg.columns.tolist()]
            )
            card_agg = card_agg.reset_index()

            # ==========================================================
            # STEP 2: AGGREGATE TO CLIENT LEVEL (SK_ID_CURR)
            # ==========================================================
            client_level_aggs = {
                "SK_ID_PREV": ["count"],  # Total number of credit cards the client has
                "CARD_MONTHS_BALANCE_SIZE": ["sum"],  # Total months of CC history
                "CARD_CC_UTILIZATION_MAX": [
                    "max"
                ],  # The HIGHEST utilization across ALL their cards
                "CARD_CC_UTILIZATION_MEAN": ["mean"],
                "CARD_CC_ATM_DRAWING_RATIO_MAX": ["max"],
                "CARD_CC_IS_LATE_SUM": ["sum"],
                "CARD_CC_IS_SEVERE_LATE_SUM": ["sum"],
                "CARD_AMT_DRAWINGS_ATM_CURRENT_WAS_MISSING_SUM": [
                    "sum"
                ],  # Total inactive months
            }

            final_agg = card_agg.groupby("SK_ID_CURR").agg(client_level_aggs)
            final_agg.columns = pd.Index(
                [f"CLIENT_{e[0]}_{e[1].upper()}" for e in final_agg.columns.tolist()]
            )

            logger.info(
                f"Successfully engineered Two-Step Credit Card proxies. Shape: {final_agg.shape}"
            )
            return final_agg.reset_index()

        except Exception as e:
            logger.error(f"Failed to aggregate Credit Card data: {e}")
            raise


class InstallmentsAggregator(FeatureAggregator):
    """Handles the 1:N:M relationship for internal Installment Payments using Two-Step Aggregation."""

    def aggregate(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(
                "Aggregating installments_payments.csv (Two-Step Aggregation)..."
            )
            df_inst = inst_df.copy()

            # --- 1. HANDLE MISSING DATA (The "Never Paid" Signal) ---
            df_inst["IS_MISSED_PAYMENT"] = df_inst["AMT_PAYMENT"].isna().astype(int)
            df_inst["AMT_PAYMENT"] = df_inst["AMT_PAYMENT"].fillna(0.0)

            # If they missed the payment entirely, artificially set the entry date
            # to be 365 days LATE. This ensures PAYMENT_DELAY mathematically reflects a severe default.
            df_inst["DAYS_ENTRY_PAYMENT"] = df_inst["DAYS_ENTRY_PAYMENT"].fillna(
                df_inst["DAYS_INSTALMENT"] + 365
            )

            # --- 2. ENGINEER BEHAVIORAL FEATURES ---
            # Payment Delay: Positive = Late, Negative = Early
            df_inst["PAYMENT_DELAY"] = (
                df_inst["DAYS_ENTRY_PAYMENT"] - df_inst["DAYS_INSTALMENT"]
            )
            df_inst["IS_LATE"] = (df_inst["PAYMENT_DELAY"] > 0).astype(int)
            df_inst["IS_SEVERE_LATE"] = (df_inst["PAYMENT_DELAY"] > 30).astype(int)

            # Payment Fraction: Paid / Prescribed
            # Replace 0 instalment with NaN to prevent divide-by-zero Infinity
            df_inst["SAFE_INSTALMENT"] = df_inst["AMT_INSTALMENT"].replace(
                0, float("nan")
            )
            df_inst["PAYMENT_FRACTION"] = (
                df_inst["AMT_PAYMENT"] / df_inst["SAFE_INSTALMENT"]
            )

            # Guardrails: Clip astronomical fractions (e.g., prepaying a massive loan)
            df_inst["PAYMENT_FRACTION"] = df_inst["PAYMENT_FRACTION"].clip(upper=5.0)
            df_inst["IS_UNDERPAID"] = (df_inst["PAYMENT_FRACTION"] < 0.99).astype(
                int
            )  # 0.99 handles rounding errors

            # ==========================================================
            # STEP 1: AGGREGATE TO LOAN LEVEL (SK_ID_PREV)
            # ==========================================================
            loan_level_aggs = {
                "NUM_INSTALMENT_VERSION": [
                    "nunique"
                ],  # Count how many times the loan terms were changed
                "PAYMENT_DELAY": ["max", "mean", "sum"],
                "PAYMENT_FRACTION": ["mean", "min"],
                "IS_LATE": ["sum", "mean"],
                "IS_SEVERE_LATE": ["sum"],
                "IS_UNDERPAID": ["sum"],
                "IS_MISSED_PAYMENT": ["sum"],
            }

            loan_agg = df_inst.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
                loan_level_aggs
            )
            loan_agg.columns = pd.Index(
                [f"LOAN_{e[0]}_{e[1].upper()}" for e in loan_agg.columns.tolist()]
            )
            loan_agg = loan_agg.reset_index()

            # ==========================================================
            # STEP 2: AGGREGATE TO CLIENT LEVEL (SK_ID_CURR)
            # ==========================================================
            client_level_aggs = {
                "SK_ID_PREV": [
                    "count"
                ],  # Total previous loans with installment history
                "LOAN_NUM_INSTALMENT_VERSION_NUNIQUE": [
                    "sum"
                ],  # Total term restructures across all loans
                "LOAN_PAYMENT_DELAY_MAX": [
                    "max"
                ],  # The absolute worst delay across any loan
                "LOAN_PAYMENT_DELAY_MEAN": ["mean"],
                "LOAN_PAYMENT_FRACTION_MIN": [
                    "min"
                ],  # The worst underpayment fraction across any loan
                "LOAN_IS_LATE_SUM": ["sum"],
                "LOAN_IS_SEVERE_LATE_SUM": ["sum"],
                "LOAN_IS_UNDERPAID_SUM": ["sum"],
                "LOAN_IS_MISSED_PAYMENT_SUM": ["sum"],
            }

            final_agg = loan_agg.groupby("SK_ID_CURR").agg(client_level_aggs)
            final_agg.columns = pd.Index(
                [
                    f"CLIENT_INST_{e[0]}_{e[1].upper()}"
                    for e in final_agg.columns.tolist()
                ]
            )

            logger.info(
                f"Successfully engineered Two-Step Installment proxies. Shape: {final_agg.shape}"
            )
            return final_agg.reset_index()

        except Exception as e:
            logger.error(f"Failed to aggregate Installments data: {e}")
            raise


class PreviousApplicationAggregator(FeatureAggregator):
    """Handles the 1:N relationship for the internal previous application history."""

    def aggregate(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(
                "Aggregating previous_application.csv (Internal Application History)..."
            )
            df_prev = prev_df.copy()

            # --- 1. HANDLE SYSTEM ANOMALIES & EXPLICIT FLAGS ---
            # Temporal anomalies (The 1000-year trap)
            days_cols = [
                "DAYS_FIRST_DRAWING",
                "DAYS_FIRST_DUE",
                "DAYS_LAST_DUE_1ST_VERSION",
                "DAYS_LAST_DUE",
                "DAYS_TERMINATION",
            ]
            for col in days_cols:
                if col in df_prev.columns:
                    df_prev[col] = df_prev[col].replace(365243, float("nan"))

            # Area anomalies (-1 means unknown)
            df_prev["AREA_WAS_NEGATIVE"] = (df_prev["SELLERPLACE_AREA"] == -1).astype(
                int
            )
            df_prev["SELLERPLACE_AREA"] = df_prev["SELLERPLACE_AREA"].replace(
                -1, float("nan")
            )

            # --- 2. DEEPSEEK UPGRADE: PRE-AGGREGATION MISSINGNESS FLAGS ---
            # Capture missingness as a behavior BEFORE aggregating
            df_prev["HAS_DISBURSEMENT_INFO"] = (
                df_prev["DAYS_FIRST_DRAWING"].notna().astype(int)
            )
            df_prev["HAS_DOWN_PAYMENT"] = (
                df_prev["AMT_DOWN_PAYMENT"].notna().astype(int)
            )
            df_prev["HAS_INTEREST_RATE"] = (
                df_prev["RATE_INTEREST_PRIMARY"].notna().astype(int)
            )

            # --- 3. BEHAVIORAL FEATURE: Application Status Ratios ---
            df_prev["PREV_IS_APPROVED"] = (
                df_prev["NAME_CONTRACT_STATUS"] == "Approved"
            ).astype(int)
            df_prev["PREV_IS_REFUSED"] = (
                df_prev["NAME_CONTRACT_STATUS"] == "Refused"
            ).astype(int)
            df_prev["PREV_IS_CANCELED"] = (
                df_prev["NAME_CONTRACT_STATUS"] == "Canceled"
            ).astype(int)

            # --- 4. BEHAVIORAL FEATURE: The "Credit Ask" Gap ---
            df_prev["SAFE_CREDIT"] = df_prev["AMT_CREDIT"].replace(0, float("nan"))
            df_prev["APPLICATION_CREDIT_RATIO"] = (
                df_prev["AMT_APPLICATION"] / df_prev["SAFE_CREDIT"]
            )
            df_prev["APPLICATION_CREDIT_RATIO"] = df_prev[
                "APPLICATION_CREDIT_RATIO"
            ].clip(upper=5.0)

            # --- 5. PRODUCT INTEREST RISK ---
            df_prev["YIELD_HIGH"] = (df_prev["NAME_YIELD_GROUP"] == "high").astype(int)

            # ==========================================================
            # AGGREGATE TO CLIENT LEVEL (SK_ID_CURR)
            # ==========================================================
            aggregations = {
                "SK_ID_PREV": ["count"],  # Total number of times they applied
                "AMT_APPLICATION": ["min", "max", "mean"],
                "AMT_CREDIT": ["min", "max", "mean"],
                "AMT_DOWN_PAYMENT": ["max", "mean"],
                "APPLICATION_CREDIT_RATIO": ["max", "mean"],
                "PREV_IS_APPROVED": ["sum", "mean"],
                "PREV_IS_REFUSED": ["sum", "mean"],
                "PREV_IS_CANCELED": ["sum", "mean"],
                "YIELD_HIGH": ["sum", "mean"],
                "DAYS_DECISION": ["min", "max", "mean"],
                "CNT_PAYMENT": ["max", "mean"],
                "NFLAG_INSURED_ON_APPROVAL": ["sum", "mean"],
                # New Pre-Aggregation Flags
                "AREA_WAS_NEGATIVE": ["mean"],
                "HAS_DISBURSEMENT_INFO": ["mean"],
                "HAS_DOWN_PAYMENT": ["mean"],
                "HAS_INTEREST_RATE": ["mean"],
                "SELLERPLACE_AREA": ["max", "median"],
            }

            agg_df = df_prev.groupby("SK_ID_CURR").agg(aggregations)
            agg_df.columns = pd.Index(
                [f"PREV_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()]
            )

            logger.info(
                f"Successfully engineered Previous Application proxies. Shape: {agg_df.shape}"
            )
            return agg_df.reset_index()

        except Exception as e:
            logger.error(f"Failed to aggregate Previous Application data: {e}")
            raise


# ---------------------------------------------------------
# 3. THE MERGER (Single Responsibility)
# ---------------------------------------------------------
class FeatureMerger:
    """Strictly handles the merging of multiple datasets onto the main spine."""

    @staticmethod
    def merge(app_df: pd.DataFrame, features_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        try:
            merged_df = app_df.copy()
            for feature_df in features_dfs:
                logger.info(
                    f"Merging feature table with {feature_df.shape[1] - 1} new columns..."
                )
                merged_df = merged_df.merge(feature_df, on="SK_ID_CURR", how="left")
            logger.info(f"Merge complete. Final dataset shape: {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Failed to merge features: {e}")
            raise
