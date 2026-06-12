import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger

# Specific ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import shap

from config import CONFIG
from preprocess import BaselinePreprocessor

# ---------------------------------------------------------
# 1. THE INTERFACE (Abstract Base Class)
# ---------------------------------------------------------
class BaseCreditModel(ABC):
    """
    The unified interface for all ML models in the pipeline.
    Forces child classes to implement a standard `train` method.
    """
    def __init__(self, random_state: int = CONFIG.pipeline.random_state):
        self.random_state = random_state
        self.model = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """The standard training signature all models must accept."""
        pass

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Unified prediction interface."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save_pipeline(self, filepath: str, preprocessor_obj: BaselinePreprocessor, selector_obj=None) -> None:
        """
        DRY Implementation: Handles saving the model and its dependencies.
        """
        try:
            logger.info(f"Saving inference pipeline to {filepath}...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            pipeline_data = {
                'preprocessor': preprocessor_obj.preprocessor,
                'feature_names': preprocessor_obj.feature_names,
                'selector': getattr(selector_obj, 'selector', None) if selector_obj else None,
                'selected_feature_names': getattr(selector_obj, 'selected_feature_names', None) if selector_obj else None,
                'model': self.model
            }
            joblib.dump(pipeline_data, filepath)
            logger.info("Successfully saved pipeline.")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise


# ---------------------------------------------------------
# 2. THE COMPLIANT MODEL (Logistic Regression)
# ---------------------------------------------------------
class LogisticRegressionModel(BaseCreditModel):
    """Transparent baseline for regulatory environments."""

    def __init__(self, random_state: int = CONFIG.lr.random_state) -> None:
        super().__init__(random_state)
        self.model = LogisticRegression(
            class_weight='balanced', 
            max_iter=CONFIG.lr.max_iter, 
            random_state=self.random_state
        )
        logger.info("Initialized LogisticRegressionModel.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        logger.info("Training Logistic Regression Model...")
        self.model.fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            logger.info(f"Validation AUC: {val_auc:.4f}")

    def extract_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """Unique to linear models: Extracts coefficients for interpretation."""
        coefficients = self.model.coef_[0]
        return pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', key=abs, ascending=False)


# ---------------------------------------------------------
# 3. THE ADVANCED MODEL (LightGBM)
# ---------------------------------------------------------
class GradientBoostingModel(BaseCreditModel):
    """High-performance non-linear model with SHAP capabilities."""

    def __init__(self, random_state: int = CONFIG.lgb.random_state) -> None:
        super().__init__(random_state)
        self.model = lgb.LGBMClassifier(
            n_estimators=CONFIG.lgb.n_estimators,
            learning_rate=CONFIG.lgb.learning_rate,
            class_weight=CONFIG.lgb.class_weight,
            max_depth=CONFIG.lgb.max_depth,
            subsample=CONFIG.lgb.subsample,
            colsample_bytree=CONFIG.lgb.colsample_bytree,
            random_state=self.random_state,
            n_jobs=CONFIG.lgb.n_jobs,
            verbose=CONFIG.lgb.verbose
        )
        logger.info("Initialized GradientBoostingModel (LightGBM).")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        logger.info("Training LightGBM model with early stopping...")
        os.makedirs("results/model", exist_ok=True)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ]
        )

        # Generate Learning Curves automatically
        plt.figure(figsize=(10, 6))
        lgb.plot_metric(self.model, metric='auc')
        plt.title('LightGBM Learning Curve')
        plt.tight_layout()
        plt.savefig('results/model/learning_curves.png')
        plt.close('all')

    def generate_global_shap(self, X: pd.DataFrame) -> None:
        """Unique to Tree models: Global SHAP interpretation."""
        logger.info("Calculating global SHAP values...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig('results/model/global_feature_importance.png')
        plt.close('all')
    
    def generate_local_shap(self, X_client: pd.DataFrame, client_id: str, prefix: str) -> None:
        """Unique to Tree models: Local SHAP interpretation for individual clients."""
        try:
            logger.info(f"Generating local SHAP force plot for {prefix} client {client_id}...")
            os.makedirs("results/clients_outputs", exist_ok=True)
            
            # Note: SHAP requires JS for interactive force plots, so we save them as HTML files
            shap.initjs()
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_client)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                expected_value = explainer.expected_value[1]
            else:
                expected_value = explainer.expected_value

            force_plot = shap.force_plot(
                expected_value, 
                shap_values[0, :], 
                X_client.iloc[0, :]
            )
            
            filepath = f"results/clients_outputs/{prefix}_client_{client_id}_force_plot.html"
            shap.save_html(filepath, force_plot)
            logger.info(f"Saved local SHAP report to {filepath}")

        except Exception as e:
            logger.error(f"Failed to generate local SHAP: {e}")
            raise