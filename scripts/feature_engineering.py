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
            if 'DAYS_CREDIT' in df_working.columns:
                df_working['RECENT_LOAN_FLAG'] = (df_working['DAYS_CREDIT'] > -180).astype(int)
            
            # --- BEHAVIORAL FEATURE 2: Overdue History ---
            if 'AMT_CREDIT_MAX_OVERDUE' in df_working.columns:
                df_working['HAS_BEEN_OVERDUE'] = (df_working['AMT_CREDIT_MAX_OVERDUE'] > 0).astype(int)
            
            # System Placeholder Outlier Detection
            if 'DAYS_CREDIT_ENDDATE' in df_working.columns:
                # Flag revolving/open-ended credit (System anomalies like 31,199 days)
                df_working['IS_OPEN_ENDED_CREDIT'] = (df_working['DAYS_CREDIT_ENDDATE'] > 10000).astype(int)
                # Replace extreme outlier with NaN so it doesn't wreck downstream metrics
                df_working.loc[df_working['DAYS_CREDIT_ENDDATE'] > 10000, 'DAYS_CREDIT_ENDDATE'] = float('nan')

            aggregations = {
                'SK_ID_BUREAU': ['count'],
                'DAYS_CREDIT': ['mean', 'min', 'max'],
                'AMT_CREDIT_SUM': ['sum', 'mean'],
                'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
            }
            
            if 'DAYS_CREDIT_ENDDATE' in df_working.columns:
                aggregations['DAYS_CREDIT_ENDDATE'] = ['mean', 'max']
            if 'IS_OPEN_ENDED_CREDIT' in df_working.columns:
                aggregations['IS_OPEN_ENDED_CREDIT'] = ['sum', 'max']
            if 'RECENT_LOAN_FLAG' in df_working.columns:
                aggregations['RECENT_LOAN_FLAG'] = ['sum']
            if 'HAS_BEEN_OVERDUE' in df_working.columns:
                aggregations['HAS_BEEN_OVERDUE'] = ['max']
                
            # If bureau balance features have been merged in, roll them up to SK_ID_CURR
            bal_cols = [c for c in df_working.columns if c.startswith('BUREAU_BAL_')]
            for col in bal_cols:
                aggregations[col] = ['mean', 'sum']
                
            agg_df = df_working.groupby('SK_ID_CURR').agg(aggregations)
            agg_df.columns = pd.Index([f"BUREAU_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()])
            
            if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in agg_df and 'BUREAU_AMT_CREDIT_SUM_SUM' in agg_df:
                agg_df['BUREAU_DEBT_CREDIT_RATIO'] = (
                    agg_df['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
                    agg_df['BUREAU_AMT_CREDIT_SUM_SUM'].replace(0, float('nan'))
                )
                
            # --- BEHAVIORAL FEATURE 3: Active Debt Ratio ---
            if 'CREDIT_ACTIVE' in df_working.columns:
                active_df = df_working[df_working['CREDIT_ACTIVE'] == 'Active']
                active_agg = active_df.groupby('SK_ID_CURR').agg({
                    'AMT_CREDIT_SUM': ['sum'],
                    'AMT_CREDIT_SUM_DEBT': ['sum']
                })
                active_agg.columns = ['BUREAU_ACTIVE_CREDIT_SUM', 'BUREAU_ACTIVE_DEBT_SUM']
                
                active_agg['BUREAU_ACTIVE_DEBT_RATIO'] = (
                    active_agg['BUREAU_ACTIVE_DEBT_SUM'] / 
                    active_agg['BUREAU_ACTIVE_CREDIT_SUM'].replace(0, float('nan'))
                )
                
                agg_df = agg_df.join(active_agg[['BUREAU_ACTIVE_DEBT_RATIO']], how='left')

            logger.info(f"Successfully engineered Bureau proxies. Shape: {agg_df.shape}")
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
            df_working['IS_APPROVED'] = (df_working['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
            df_working['IS_REFUSED'] = (df_working['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
            
            aggregations = {
                'SK_ID_PREV': ['count'],
                'AMT_APPLICATION': ['mean', 'max'],
                'IS_APPROVED': ['mean', 'sum'],
                'IS_REFUSED': ['mean', 'sum'],
            }
            agg_df = df_working.groupby('SK_ID_CURR').agg(aggregations)
            agg_df.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()])
            
            logger.info(f"Successfully engineered Previous App proxies. Shape: {agg_df.shape}")
            return agg_df.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate previous application data: {e}")
            raise


class POSCashBalanceAggregator(FeatureAggregator):
    """Handles specific business logic for POS/Cash balance data."""
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating POS_CASH_balance.csv (Internal POS/Cash History)...")
            
            df_working = df.copy()
            
            aggregations = {
                'SK_ID_PREV': ['count'], # Total months of POS loans
                'SK_DPD': ['max', 'mean'], # Severe delinquency
                'SK_DPD_DEF': ['max', 'mean'], # Severe delinquency (with tolerance)
                'CNT_INSTALMENT': ['max', 'mean'], # Term of loan
                'CNT_INSTALMENT_FUTURE': ['max', 'mean'] # Remaining term
            }
            
            # Aggregate to the client level directly (since SK_ID_CURR is present)
            agg_df = df_working.groupby('SK_ID_CURR').agg(aggregations)
            agg_df.columns = pd.Index([f"POS_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()])
            
            logger.info(f"Successfully engineered POS/Cash Balance proxies. Shape: {agg_df.shape}")
            return agg_df.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate POS Cash balance data: {e}")
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
                logger.info(f"Merging feature table with {feature_df.shape[1] - 1} new columns...")
                merged_df = merged_df.merge(feature_df, on='SK_ID_CURR', how='left')
            logger.info(f"Merge complete. Final dataset shape: {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Failed to merge features: {e}")
            raise