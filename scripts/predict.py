import pandas as pd
import joblib
import os
import numpy as np
from loguru import logger

class CreditRiskPredictor:
    """
    Loads a saved econometric pipeline and generates predictions.
    Architected to support both bulk batch predictions (Kaggle) and real-time inference (APIs).
    """

    def __init__(self, pipeline_path: str) -> None:
        try:
            logger.info(f"Loading inference pipeline from {pipeline_path}...")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
                
            pipeline_data = joblib.load(pipeline_path)
            
            # Reconstruct the pipeline state
            self.preprocessor = pipeline_data['preprocessor']
            self.feature_names = pipeline_data['feature_names']
            self.selector = pipeline_data.get('selector')
            self.selected_feature_names = pipeline_data.get('selected_feature_names')
            self.model = pipeline_data['model']
            
            logger.info("Successfully loaded preprocessor, selector, and model.")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Core inference logic. Can be used for a single customer (API) or millions (Batch).
        
        Args:
            X (pd.DataFrame): Raw feature data (must NOT contain the target or ID columns).
            
        Returns:
            np.ndarray: Array of default probabilities.
        """
        try:
            logger.info("Applying preprocessor transformations...")
            # 1. Transform categorical/numerical features
            X_processed = self.preprocessor.transform(X)
            X_df = pd.DataFrame(X_processed, columns=self.feature_names)
            
            # 2. Apply feature selection if the pipeline was trained with RFE
            if self.selector is not None and self.selected_feature_names is not None:
                logger.info("Applying RFE feature selection filter...")
                X_filtered_np = self.selector.transform(X_df)
                X_final = pd.DataFrame(X_filtered_np, columns=self.selected_feature_names)
            else:
                X_final = X_df
            
            # 3. Generate probabilities
            logger.info("Generating default probabilities...")
            probabilities = self.model.predict_proba(X_final)[:, 1]
            return probabilities
            
        except Exception as e:
            logger.error(f"Failed during inference prediction: {e}")
            raise

    def generate_kaggle_submission(self, df_test: pd.DataFrame, id_col: str, output_csv_path: str) -> None:
        """
        Specific utility for Kaggle batch submissions.
        """
        try:
            logger.info("Preparing test data for Kaggle submission...")
            
            # Isolate IDs and Features
            client_ids = df_test[id_col]
            X_test = df_test.drop(columns=[id_col])
            
            # Utilize the core predict method
            probabilities = self.predict(X_test)
            
            # Format and Save
            submission_df = pd.DataFrame({
                id_col: client_ids,
                'TARGET': probabilities
            })
            
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            submission_df.to_csv(output_csv_path, index=False)
            logger.info(f"Successfully saved Kaggle submission to {output_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate Kaggle submission: {e}")
            raise