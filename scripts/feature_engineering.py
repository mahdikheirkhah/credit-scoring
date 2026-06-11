import pandas as pd
from loguru import logger

class RelationalFeatureEngineer:
    """Handles the aggregation of 'one-to-many' relational tables."""

    def __init__(self) -> None:
        try:
            logger.info("Initialized RelationalFeatureEngineer.")
        except Exception as e:
            logger.error(f"Failed to initialize RelationalFeatureEngineer: {e}")
            raise

    def aggregate_bureau(self, bureau_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating bureau.csv (External Credit History)...")
            aggregations = {
                'SK_ID_BUREAU': ['count'],
                'DAYS_CREDIT': ['mean', 'min'],
                'AMT_CREDIT_SUM': ['sum'],
                'AMT_CREDIT_SUM_DEBT': ['sum'],
            }
            bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(aggregations)
            bureau_agg.columns = pd.Index([f"BUREAU_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns.tolist()])
            
            if 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM' in bureau_agg and 'BUREAU_AMT_CREDIT_SUM_SUM' in bureau_agg:
                bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
                    bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / 
                    bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'].replace(0, float('nan'))
                )
            logger.info(f"Successfully engineered Bureau proxies. Shape: {bureau_agg.shape}")
            return bureau_agg.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate bureau data: {e}")
            raise

    def aggregate_previous_applications(self, prev_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Aggregating previous_application.csv (Internal History)...")
            prev_df['IS_APPROVED'] = (prev_df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
            prev_df['IS_REFUSED'] = (prev_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
            
            aggregations = {
                'SK_ID_PREV': ['count'],
                'AMT_APPLICATION': ['mean', 'max'],
                'IS_APPROVED': ['mean', 'sum'],
                'IS_REFUSED': ['mean', 'sum'],
            }
            prev_agg = prev_df.groupby('SK_ID_CURR').agg(aggregations)
            prev_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns.tolist()])
            logger.info(f"Successfully engineered Previous App proxies. Shape: {prev_agg.shape}")
            return prev_agg.reset_index()
        except Exception as e:
            logger.error(f"Failed to aggregate previous application data: {e}")
            raise

    def merge_features(self, app_df: pd.DataFrame, new_features_list: list[pd.DataFrame]) -> pd.DataFrame:
        try:
            merged_df = app_df.copy()
            for feature_df in new_features_list:
                logger.info(f"Merging feature table with {feature_df.shape[1] - 1} new columns...")
                merged_df = merged_df.merge(feature_df, on='SK_ID_CURR', how='left')
            logger.info(f"Merge complete. Final dataset shape: {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Failed to merge features: {e}")
            raise