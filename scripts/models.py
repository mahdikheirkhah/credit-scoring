import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
# Specific ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import shap
from sklearn.tree import DecisionTreeClassifier

from scripts.config import CONFIG
from scripts.preprocess import BaselinePreprocessor


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
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """The standard training signature all models must accept."""
        pass

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Unified prediction interface."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save_pipeline(
        self, filepath: str, preprocessor_obj: BaselinePreprocessor, selector_obj=None
    ) -> None:
        """
        DRY Implementation: Handles saving the model and its dependencies.

        Args:
            filepath (str): The destination path for the saved pipeline pickle file.
            preprocessor_obj (BaselinePreprocessor): The stateful preprocessor object to save.
            selector_obj (Optional): The feature selector object used, if any.

        Returns:
            None
        """
        try:
            logger.info(f"Saving inference pipeline to {filepath}...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            pipeline_data = {
                # ---> THE FIX: Save the entire custom wrapper object <---
                "preprocessor_obj": preprocessor_obj,
                "selector": (
                    getattr(selector_obj, "selector", None) if selector_obj else None
                ),
                "selected_feature_names": (
                    getattr(selector_obj, "selected_feature_names", None)
                    if selector_obj
                    else None
                ),
                "model": self.model,
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
            class_weight="balanced",
            max_iter=CONFIG.lr.max_iter,
            random_state=self.random_state,
        )
        logger.info("Initialized LogisticRegressionModel.")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        logger.info("Training Logistic Regression Model...")
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            logger.info(f"Validation AUC: {val_auc:.4f}")

    def extract_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """Unique to linear models: Extracts coefficients for interpretation."""
        coefficients = self.model.coef_[0]
        return pd.DataFrame(
            {"Feature": feature_names, "Coefficient": coefficients}
        ).sort_values(by="Coefficient", key=abs, ascending=False)


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
            verbose=CONFIG.lgb.verbose,
            linear_tree=CONFIG.lgb.linear_tree
        )
        logger.info("Initialized GradientBoostingModel (LightGBM).")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        logger.info("Training LightGBM model with early stopping...")
        os.makedirs("results/model", exist_ok=True)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=CONFIG.lgb.early_stopping_rounds), lgb.log_evaluation(CONFIG.lgb.early_stopping_rounds)],
        )

        # Generate Learning Curves automatically
        plt.figure(figsize=(10, 6))
        lgb.plot_metric(self.model, metric="auc")
        plt.title("LightGBM Learning Curve")
        plt.tight_layout()
        plt.savefig("results/model/learning_curves.png")
        plt.close("all")

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
        plt.savefig("results/model/global_feature_importance.png")
        plt.close("all")

    def generate_local_shap(
        self, X_client: pd.DataFrame, client_id: str, prefix: str
    ) -> None:
        """Unique to Tree models: Local SHAP interpretation for individual clients."""
        try:
            logger.info(
                f"Generating local SHAP force plot for {prefix} client {client_id}..."
            )
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
                expected_value, shap_values[0, :], X_client.iloc[0, :]
            )

            filepath = (
                f"results/clients_outputs/{prefix}_client_{client_id}_force_plot.html"
            )
            shap.save_html(filepath, force_plot)
            logger.info(f"Saved local SHAP report to {filepath}")

        except Exception as e:
            logger.error(f"Failed to generate local SHAP: {e}")
            raise


class PiecewiseLinearModel(BaseCreditModel):
    """
    Econometric Model Tree: Uses a Decision Tree to segment borrowers, 
    then applies a distinct Logistic Regression model to each segment (leaf).
    """

    def __init__(self, random_state: int = CONFIG.pipeline.random_state):
        super().__init__(random_state)
        
        # 1. The Global Segmenter
        self.tree = DecisionTreeClassifier(
            max_depth=CONFIG.piecewise.max_depth,
            min_samples_leaf=CONFIG.piecewise.min_samples_leaf,
            class_weight=CONFIG.piecewise.tree_class_weight,
            random_state=self.random_state
        )
        
        # 2. Dictionary to hold the local LR models
        self.leaf_models = {}
        
        # 3. Global Fallback (Used if a leaf is too pure/small to fit an LR)
        self.global_fallback = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        logger.info(f"Training Piecewise Linear Model (Max Depth: {CONFIG.piecewise.max_depth})...")
        
        # Train Global Fallback
        self.global_fallback = LogisticRegression(
            class_weight=CONFIG.piecewise.lr_class_weight, 
            solver='liblinear',
            random_state=self.random_state
        )
        self.global_fallback.fit(X_train, y_train)

        # 1. Train the Decision Tree to partition the space
        self.tree.fit(X_train, y_train)
        
        # 2. Route every training sample to its assigned Leaf ID
        leaf_indices = self.tree.apply(X_train)
        unique_leaves = np.unique(leaf_indices)
        
        logger.info(f"Tree created {len(unique_leaves)} distinct borrower segments.")

        # 3. Train a separate LR for each leaf
        for leaf in unique_leaves:
            # Subset the data falling into this specific leaf
            mask = (leaf_indices == leaf)
            X_leaf, y_leaf = X_train[mask], y_train[mask]
            
            # EDGE CASE: If the leaf has only 1 class (e.g., 0 defaults), LR will fail.
            if len(np.unique(y_leaf)) < 2:
                logger.warning(f"Leaf {leaf} contains only 1 class. Assigning global fallback model.")
                self.leaf_models[leaf] = self.global_fallback
                continue
                
            # Fit local econometric model
            lr = LogisticRegression(
                class_weight=CONFIG.piecewise.lr_class_weight,
                C=CONFIG.piecewise.lr_c_value,
                solver="liblinear",
                max_iter=CONFIG.piecewise.lr_max_iter,
                random_state=self.random_state
            )
            lr.fit(X_leaf, y_leaf)
            self.leaf_models[leaf] = lr
            
        logger.info("Successfully trained all local econometric models.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Routes unseen data through the tree, then applies the specific leaf's LR model."""
        if not self.leaf_models:
            raise ValueError("Model has not been trained yet.")
            
        # Route new clients to their leaves
        leaf_indices = self.tree.apply(X)
        probabilities = np.zeros(len(X))
        
        # Apply the specific LR equation for each leaf segment
        for leaf in np.unique(leaf_indices):
            mask = (leaf_indices == leaf)
            
            # If we see a leaf in inference that we didn't train on (rare but possible), use fallback
            model = self.leaf_models.get(leaf, self.global_fallback)
            
            probabilities[mask] = model.predict_proba(X[mask])[:, 1]
            
        return probabilities
        
    def extract_leaf_coefficients(self, feature_names: list) -> dict:
        """Returns the specific regression equations for regulatory audits."""
        audit_dict = {}
        for leaf, model in self.leaf_models.items():
            if model != self.global_fallback:
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_[0]
                }).sort_values(by='Coefficient', key=abs, ascending=False)
                audit_dict[leaf] = coef_df
        return audit_dict

    # UPDATE IN scripts/models.py inside the PiecewiseLinearModel class

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Routes unseen data through the tree, then applies the specific leaf's LR model."""
        if not self.leaf_models:
            raise ValueError("Model has not been trained yet.")
            
        leaf_indices = self.tree.apply(X)
        
        # ---> API FIX: Create a 2D array to mimic Scikit-Learn exactly <---
        # Shape: (number of samples, 2 classes)
        probabilities = np.zeros((len(X), 2))
        
        for leaf in np.unique(leaf_indices):
            mask = (leaf_indices == leaf)
            model = self.leaf_models.get(leaf, self.global_fallback)
            
            # Predict returns a 2D array of [Prob_Good, Prob_Default]. We store the whole thing.
            probabilities[mask] = model.predict_proba(X[mask])
            
        return probabilities

    def save_pipeline(self, filepath: str, preprocessor_obj, selector_obj=None) -> None:
        """
        ---> OOP FIX: Overrides the BaseCreditModel save method <---
        Saves the ENTIRE wrapper object (self) because our logic is split 
        across self.tree and self.leaf_models, rather than a single self.model.
        """
        import joblib
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Safely handle the selected features if RFE was used
        selected_features = None
        if selector_obj is not None and hasattr(selector_obj, 'selected_feature_names'):
            selected_features = selector_obj.selected_feature_names
            
        pipeline_data = {
            "preprocessor_obj": preprocessor_obj,
            "selector": selector_obj,
            "selected_feature_names": selected_features,
            "model": self  # Save the entire Piecewise class instance!
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Successfully saved Piecewise pipeline to {filepath}")