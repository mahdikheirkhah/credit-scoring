import pandas as pd
from loguru import logger
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from config import CONFIG

class RFEFeatureSelector:
    """Handles mathematically robust feature reduction."""

    def __init__(self, random_state: int = CONFIG.lr.random_state) -> None:
        self.random_state = random_state
        self.selector = None
        self.selected_feature_names = None

    def select(self, X: pd.DataFrame, y: pd.Series, n_features: int = CONFIG.lr.rfe_n_features) -> pd.DataFrame:
        """Fits RFE and reduces the dataset to the top n_features."""
        try:
            logger.info(f"Starting RFE to select top {n_features} features...")
            
            estimator = LogisticRegression(
                class_weight='balanced', 
                C=CONFIG.lr.rfe_c_value, 
                max_iter=CONFIG.lr.rfe_max_iter, 
                random_state=self.random_state
            )
            
            self.selector = RFE(
                estimator=estimator, 
                n_features_to_select=n_features, 
                step=CONFIG.lr.rfe_step
            )
            
            X_selected = self.selector.fit_transform(X, y)
            
            selected_columns = X.columns[self.selector.support_]
            self.selected_feature_names = selected_columns.tolist()
            
            logger.info(f"RFE complete. Reduced to {len(self.selected_feature_names)} features.")
            return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
            
        except Exception as e:
            logger.error(f"Failed during RFE feature selection: {e}")
            raise