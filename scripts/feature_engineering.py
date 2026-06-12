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
            aggregations = {
                'SK_ID_BUREAU': ['count'],
                'DAYS_CREDIT': ['mean', 'min'],
                'AMT_CREDIT_SUM': ['sum'],
                'AMT_CREDIT_SUM_DEBT': ['sum'],
            }
            agg_df = df.groupby('SK_ID_CURR').agg(aggregations)
            agg_df.columns = pd.Index([f"BUREAU_{e[0]}_{e[1].upper()}" for e in agg_df.columns.tolist()])
            
            if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in agg_df and 'BUREAU_AMT_CREDIT_SUM_SUM' in agg_df:
                agg_df['BUREAU_DEBT_CREDIT_RATIO'] = (
                    agg_df['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
                    agg_df['BUREAU_AMT_CREDIT_SUM_SUM'].replace(0, float('nan'))
                )
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