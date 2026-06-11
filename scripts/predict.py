import pandas as pd
import joblib
import os
from loguru import logger
from feature_engineering import RelationalFeatureEngineer  # <-- NEW IMPORT


class KagglePredictor:
    """
    Loads a saved econometric pipeline and generates predictions on unseen test data.
    """

    def __init__(self, pipeline_path: str) -> None:
        """
        Initializes the predictor by loading the saved preprocessor, RFE selector, and model.
        
        Args:
            pipeline_path (str): Path to the saved .pkl file.
        """
        try:
            logger.info(f"Loading inference pipeline from {pipeline_path}...")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
                
            pipeline_data = joblib.load(pipeline_path)
            
            # 1. Load the Preprocessor (133 columns)
            self.preprocessor = pipeline_data['preprocessor']
            self.feature_names = pipeline_data['feature_names']
            
            # 2. Load the RFE Selector (Filters down to 30 columns)
            self.selector = pipeline_data.get('selector')
            self.selected_feature_names = pipeline_data.get('selected_feature_names')
            
            # 3. Load the Model (Expects 30 columns)
            self.model = pipeline_data['model']
            
            logger.info("Successfully loaded preprocessor, selector, and model.")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise

    def generate_submission(self, df_test: pd.DataFrame, output_csv_path: str) -> None:
        """
        Processes the engineered test data, predicts probabilities, and saves the submission.
        """
        try:
            logger.info("Preparing test data for prediction...")
            
            # 1. Isolate the Customer ID
            client_ids = df_test['SK_ID_CURR']
            X_test = df_test.drop(columns=['SK_ID_CURR'])
            
            # 2. Transform the data (Outputs 133 columns)
            logger.info("Applying preprocessor transformations...")
            X_test_processed = self.preprocessor.transform(X_test)
            X_test_df = pd.DataFrame(X_test_processed, columns=self.feature_names)
            
            # 3. Apply RFE Filter
            logger.info("Applying RFE feature selection filter...")
            if self.selector is not None and self.selected_feature_names is not None:
                # Transform outputs a raw numpy array, so we must wrap it back into a DataFrame
                # using the exact 30 column names the model memorized during training.
                X_test_filtered_np = self.selector.transform(X_test_df)
                X_test_filtered = pd.DataFrame(X_test_filtered_np, columns=self.selected_feature_names)
            else:
                # Fallback in case a pipeline without RFE is loaded
                X_test_filtered = X_test_df
            
            # 4. Predict Probabilities
            logger.info("Generating default probabilities...")
            probabilities = self.model.predict_proba(X_test_filtered)[:, 1]
            
            # 5. Format and Save
            submission_df = pd.DataFrame({
                'SK_ID_CURR': client_ids,
                'TARGET': probabilities
            })
            
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            submission_df.to_csv(output_csv_path, index=False)
            logger.info(f"Successfully saved Kaggle submission to {output_csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate submission: {e}")
            raise


if __name__ == "__main__":
    try:
        # 1. Initialize our tools
        feature_engineer = RelationalFeatureEngineer()
        predictor = KagglePredictor("results/model/my_own_model.pkl")
        
        # 2. Load the raw test data AND the historical tables
        logger.info("Loading test data and historical tables...")
        df_test_raw = pd.read_csv("data/application_test.csv")
        bureau_df = pd.read_csv("data/bureau.csv")
        prev_app_df = pd.read_csv("data/previous_application.csv")
        
        # 3. Engineer the exact same proxies for the test clients
        logger.info("Engineering relational proxies for the test dataset...")
        bureau_proxies = feature_engineer.aggregate_bureau(bureau_df)
        prev_proxies = feature_engineer.aggregate_previous_applications(prev_app_df)
        
        # Merge the proxies into the test dataset
        df_test_merged = feature_engineer.merge_features(df_test_raw, [bureau_proxies, prev_proxies])

        # 4. Generate the predictions
        predictor.generate_submission(
            df_test=df_test_merged,
            output_csv_path="results/model/kaggle_submission.csv"
        )
        
    except Exception as main_e:
        logger.critical(f"Prediction script failed: {main_e}")