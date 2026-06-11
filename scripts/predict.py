import pandas as pd
import joblib
import os
from loguru import logger

class KagglePredictor:
    """Loads a saved econometric pipeline and generates predictions on unseen test data."""

    def __init__(self, pipeline_path: str) -> None:
        try:
            logger.info(f"Loading inference pipeline from {pipeline_path}...")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
                
            pipeline_data = joblib.load(pipeline_path)
            self.preprocessor = pipeline_data['preprocessor']
            self.feature_names = pipeline_data['feature_names']
            self.selector = pipeline_data.get('selector')
            self.selected_feature_names = pipeline_data.get('selected_feature_names')
            self.model = pipeline_data['model']
            logger.info("Successfully loaded preprocessor, selector, and model.")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise

    def generate_submission(self, df_test: pd.DataFrame, output_csv_path: str) -> None:
        try:
            logger.info("Preparing test data for prediction...")
            client_ids = df_test['SK_ID_CURR']
            X_test = df_test.drop(columns=['SK_ID_CURR'])
            
            logger.info("Applying preprocessor transformations...")
            X_test_processed = self.preprocessor.transform(X_test)
            X_test_df = pd.DataFrame(X_test_processed, columns=self.feature_names)
            
            logger.info("Applying RFE feature selection filter...")
            if self.selector is not None and self.selected_feature_names is not None:
                X_test_filtered_np = self.selector.transform(X_test_df)
                X_test_filtered = pd.DataFrame(X_test_filtered_np, columns=self.selected_feature_names)
            else:
                X_test_filtered = X_test_df
            
            logger.info("Generating default probabilities...")
            probabilities = self.model.predict_proba(X_test_filtered)[:, 1]
            
            submission_df = pd.DataFrame({'SK_ID_CURR': client_ids, 'TARGET': probabilities})
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            submission_df.to_csv(output_csv_path, index=False)
            logger.info(f"Successfully saved Kaggle submission to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to generate submission: {e}")
            raise