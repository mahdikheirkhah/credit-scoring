import os
import joblib
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
from loguru import logger
from preprocess import BaselinePreprocessor

class GradientBoostingModel:
    """Trains and evaluates a high-performance LightGBM model with SHAP explainability."""

    def __init__(self, random_state: int = 42) -> None:
        try:
            self.random_state = random_state
            # We initialize LightGBM with parameters optimized for imbalanced financial data
            self.model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.083,
                class_weight='balanced',
                max_depth=3,  # Kept shallow to prevent overfitting
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
            logger.info("Initialized GradientBoostingModel (LightGBM).")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def train_with_learning_curves(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Trains the model while generating learning curves to prove generalization."""
        try:
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

            # Generate Learning Curves
            logger.info("Generating learning curves...")
            plt.figure(figsize=(10, 6))
            lgb.plot_metric(self.model, metric='auc')
            plt.title('LightGBM Learning Curve (Generalization Proof)')
            plt.tight_layout()
            plt.savefig('results/model/learning_curves.png')
            plt.close('all')
            logger.info("Saved learning curves to results/model/learning_curves.png")

        except Exception as e:
            logger.error(f"Failed during training: {e}")
            raise

    def generate_global_shap(self, X: pd.DataFrame) -> None:
        """Generates the global feature importance plot using SHAP."""
        try:
            logger.info("Calculating global SHAP values...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # Handle LGBM binary classification shape differences
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False)
            plt.title('Global Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig('results/model/global_feature_importance.png')
            plt.close('all')
            logger.info("Saved global feature importance to results/model/global_feature_importance.png")

        except Exception as e:
            logger.error(f"Failed to generate global SHAP: {e}")
            raise

    def generate_local_shap(self, X_client: pd.DataFrame, client_id: str, prefix: str) -> None:
        """Generates a local force plot for a specific client to satisfy regulatory audits."""
        try:
            logger.info(f"Generating local SHAP force plot for {prefix} client {client_id}...")
            os.makedirs("results/clients_outputs", exist_ok=True)
            
            # Note: SHAP requires JS for interactive force plots, so we save them as HTML files
            # which can be opened in any browser for a highly professional presentation.
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
            logger.info("Successfully saved LightGBM pipeline.")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise