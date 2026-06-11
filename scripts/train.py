import pandas as pd
from loguru import logger
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from preprocess import BaselinePreprocessor


class EconometricBaselineModel:
    """
    Trains and evaluates a transparent Logistic Regression model for credit scoring.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes the model with class weights balanced to handle the rare default events.
        
        Args:
            random_state (int): Seed for reproducibility.
        """
        try:
            self.random_state = random_state
            # class_weight='balanced' is CRITICAL for highly imbalanced credit data
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
        """
        Evaluates the model using Stratified K-Fold Cross Validation.
        
        Args:
            X (pd.DataFrame): Preprocessed features.
            y (pd.Series): Target variable (TARGET).
            n_splits (int): Number of CV folds.
            
        Returns:
            float: The average Out-Of-Fold (OOF) AUC score.
        """
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
        """
        Extracts and maps the exact econometric coefficients for regulatory interpretability.
        
        Args:
            feature_names (list[str]): The names of the features from the preprocessor.
            
        Returns:
            pd.DataFrame: A DataFrame containing features and their corresponding weights.
        """
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

    def save_pipeline(self, preprocessor_obj: BaselinePreprocessor, filepath: str) -> None:
        """
        Saves the fitted preprocessor and trained model to disk for future inference.
        
        Args:
            preprocessor_obj (BaselinePreprocessor): The fitted preprocessor instance.
            filepath (str): The destination path for the .pkl file.
        """
        try:
            logger.info(f"Saving inference pipeline to {filepath}...")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # We bundle them together so they never get separated
            pipeline_data = {
                'preprocessor': preprocessor_obj.preprocessor,
                'feature_names': preprocessor_obj.feature_names,
                'model': self.model
            }
            joblib.dump(pipeline_data, filepath)
            logger.info("Successfully saved pipeline.")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise


if __name__ == "__main__":
    try:
        # 1. Initialize tools
        preprocessor = BaselinePreprocessor()
        baseline_model = EconometricBaselineModel()

        # 2. Load Data and inspect types
        # Note: If memory or speed is an issue during initial testing, 
        # you can pass `nrows=10000` to `pd.read_csv` inside the load_data method.
        df = preprocessor.load_data("data/application_train.csv")
        
        # Log the raw column types for the audit trail
        preprocessor.log_column_types(df)
        
        # Split target and features. We drop SK_ID_CURR so it doesn't leak into the model.
        y = df['TARGET']
        X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

        # 3. Preprocess using our new dynamic feature extractor
        numeric_cols, categorical_cols = preprocessor.get_feature_types(X)
        preprocessor.build_pipeline(numeric_cols, categorical_cols)
        X_processed = preprocessor.fit_transform(X)

        # 4. Train and Evaluate using Cross-Validation
        baseline_model.evaluate_cv(X_processed, y)

        # 5. Interpretability: Fit on full data once to get final global coefficients
        logger.info("Fitting model on the entire dataset to extract final global coefficients...")
        baseline_model.model.fit(X_processed, y)
        coef_df = baseline_model.extract_coefficients(preprocessor.feature_names)
        baseline_model.save_pipeline(preprocessor, "results/model/my_own_model.pkl")
        # Save coefficients for the audit report
        coef_df.to_csv("results/model/baseline_coefficients.csv", index=False)
        logger.info(f"Top 5 strongest predictors of default risk:\n{coef_df.head(5)}")

    except Exception as main_e:
        logger.critical(f"Pipeline execution failed: {main_e}")