import pandas as pd
from loguru import logger


class RelationalFeatureEngineer:
    """
    Handles the aggregation of 'one-to-many' relational tables into single-row 
    behavioral risk proxies for the main application dataset.
    """

    def __init__(self) -> None:
        """Initializes the Feature Engineer."""
        try:
            logger.info("Initialized RelationalFeatureEngineer.")
        except Exception as e:
            logger.error(f"Failed to initialize RelationalFeatureEngineer: {e}")
            raise

    def aggregate_bureau(self, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the external credit bureau data (history with other banks).
        
        Args:
            bureau_df (pd.DataFrame): The raw bureau dataset.
            
        Returns:
            pd.DataFrame: Aggregated bureau proxies indexed by SK_ID_CURR.
        """
        try:
            logger.info("Aggregating bureau.csv (External Credit History)...")
            
            # Create behavioral proxies
            aggregations = {
                'SK_ID_BUREAU': ['count'], # Total number of past loans
                'DAYS_CREDIT': ['mean', 'min'], # How recent are the loans?
                'AMT_CREDIT_SUM': ['sum'], # Total debt ever taken
                'AMT_CREDIT_SUM_DEBT': ['sum'], # Total current remaining debt
            }
            
            bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(aggregations)
            
            # Flatten the MultiIndex columns created by pandas .agg()
            bureau_agg.columns = pd.Index([f"BUREAU_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns.tolist()])
            
            # Engineer an advanced proxy: Current Debt to Total Credit Ratio
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
        """
        Aggregates previous applications (history with Home Credit).
        
        Args:
            prev_df (pd.DataFrame): The raw previous applications dataset.
            
        Returns:
            pd.DataFrame: Aggregated application proxies indexed by SK_ID_CURR.
        """
        try:
            logger.info("Aggregating previous_application.csv (Internal History)...")
            
            # Identify approved vs refused applications
            prev_df['IS_APPROVED'] = (prev_df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
            prev_df['IS_REFUSED'] = (prev_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
            
            aggregations = {
                'SK_ID_PREV': ['count'], # Total number of previous applications
                'AMT_APPLICATION': ['mean', 'max'], # How much do they usually ask for?
                'IS_APPROVED': ['mean', 'sum'], # Proxy: Approval rate
                'IS_REFUSED': ['mean', 'sum'],  # Proxy: Refusal rate
            }
            
            prev_agg = prev_df.groupby('SK_ID_CURR').agg(aggregations)
            prev_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns.tolist()])
            
            logger.info(f"Successfully engineered Previous App proxies. Shape: {prev_agg.shape}")
            return prev_agg.reset_index()
            
        except Exception as e:
            logger.error(f"Failed to aggregate previous application data: {e}")
            raise

    def merge_features(self, app_df: pd.DataFrame, new_features_list: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Safely merges the newly engineered proxy tables into the main application table.
        
        Args:
            app_df (pd.DataFrame): The main application dataset.
            new_features_list (list[pd.DataFrame]): List of aggregated proxy DataFrames.
            
        Returns:
            pd.DataFrame: The fully merged dataset.
        """
        try:
            merged_df = app_df.copy()
            for feature_df in new_features_list:
                logger.info(f"Merging feature table with {feature_df.shape[1] - 1} new columns...")
                # We use a LEFT join so we don't lose clients who have no history
                merged_df = merged_df.merge(feature_df, on='SK_ID_CURR', how='left')
                
            logger.info(f"Merge complete. Final dataset shape: {merged_df.shape}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to merge features: {e}")
            raise