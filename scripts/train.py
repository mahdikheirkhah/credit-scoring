import pandas as pd
import os
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from preprocess import BaselinePreprocessor

class EconometricBaselineModel:
    """Trains and evaluates a transparent Logistic Regression model."""

    def __init__(self, random_state: int = 42) -> None:
        try:
            self.random_state = random_state
            self.model = LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                random_state=self.random_state
            )
            logger.info("Initialized EconometricBaselineModel (Logistic Regression).")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def evaluate_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
        try:
            logger.info(f"Starting {n_splits}-fold Stratified CV...")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            auc_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                self.model.fit(X_train, y_train)
                y_pred_proba = self.model.predict_proba(X_val)[:, 1]
                fold_auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores.append(fold_auc)
                logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

            avg_auc = sum(auc_scores) / len(auc_scores)
            logger.info(f"Average Cross-Validation AUC: {avg_auc:.4f}")
            return avg_auc
        except Exception as e:
            logger.error(f"Failed during cross-validation: {e}")
            raise

    def extract_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        try:
            coefficients = self.model.coef_[0]
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            }).sort_values(by='Coefficient', key=abs, ascending=False)
            logger.info("Successfully extracted model coefficients for interpretability.")
            return coef_df
        except Exception as e:
            logger.error(f"Failed to extract coefficients: {e}")
            raise

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 60) -> pd.DataFrame:
        try:
            logger.info(f"Starting RFE to select top {n_features} features out of {X.shape[1]}...")
            # We add C=0.1 to force heavy regularization and prevent overfitting!
            estimator = LogisticRegression(
                class_weight='balanced', 
                C=0.1, 
                max_iter=1000, 
                random_state=self.random_state
            )
            self.selector = RFE(estimator=estimator, n_features_to_select=n_features, step=0.1)
            X_selected = self.selector.fit_transform(X, y)
            
            selected_columns = X.columns[self.selector.support_]
            self.selected_feature_names = selected_columns.tolist()
            logger.info(f"RFE complete. Maintained transparency with {len(self.selected_feature_names)} features.")
            return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        except Exception as e:
            logger.error(f"Failed during RFE feature selection: {e}")
            raise

    def save_pipeline(self, preprocessor_obj: BaselinePreprocessor, filepath: str) -> None:
        try:
            logger.info(f"Saving inference pipeline to {filepath}...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            pipeline_data = {
                'preprocessor': preprocessor_obj.preprocessor,
                'feature_names': preprocessor_obj.feature_names,
                'selector': getattr(self, 'selector', None),
                'selected_feature_names': getattr(self, 'selected_feature_names', None),
                'model': self.model
            }
            joblib.dump(pipeline_data, filepath)
            logger.info("Successfully saved pipeline.")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise