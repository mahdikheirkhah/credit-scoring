import pandas as pd
import joblib
import os
from loguru import logger


class KagglePredictor:
    """
    Loads a saved econometric pipeline and generates predictions on unseen test data.
    """

    def __init__(self, pipeline_path: str) -> None:
        """
        Initializes the predictor by loading the saved preprocessor and model.
        
        Args:
            pipeline_path (str): Path to the saved .pkl file.
        """
        try:
            logger.info(f"Loading inference pipeline from {pipeline_path}...")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
                
            pipeline_data = joblib.load(pipeline_path)
            self.preprocessor = pipeline_data['preprocessor']
            self.feature_names = pipeline_data['feature_names']
            self.model = pipeline_data['model']
            
            logger.info("Successfully loaded preprocessor and model.")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise

    def generate_submission(self, test_csv_path: str, output_csv_path: str) -> None:
        """
        Processes the test data, predicts default probabilities, and saves a Kaggle submission.
        
        Args:
            test_csv_path (str): Path to application_test.csv.
            output_csv_path (str): Path to save the final Kaggle submission.
        """
        try:
            logger.info(f"Loading test data from {test_csv_path}...")
            df_test = pd.read_csv(test_csv_path)
            
            # 1. Isolate the Customer ID for the submission
            client_ids = df_test['SK_ID_CURR']
            X_test = df_test.drop(columns=['SK_ID_CURR'])
            
            # 2. Transform the data (STRICTLY .transform(), NEVER .fit_transform() on test data)
            logger.info("Applying preprocessor transformations...")
            X_test_processed = self.preprocessor.transform(X_test)
            
            # 3. Predict Probabilities ([:, 1] gets the probability of class 1 / Default)
            logger.info("Generating default probabilities...")
            probabilities = self.model.predict_proba(X_test_processed)[:, 1]
            
            # 4. Format for Kaggle
            submission_df = pd.DataFrame({
                'SK_ID_CURR': client_ids,
                'TARGET': probabilities
            })
            
            # 5. Save output
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            submission_df.to_csv(output_csv_path, index=False)
            logger.info(f"Successfully saved Kaggle submission to {output_csv_path}")
            logger.info(f"Submission preview:\n{submission_df.head()}")
            
        except Exception as e:
            logger.error(f"Failed to generate submission: {e}")
            raise


if __name__ == "__main__":
    try:
        predictor = KagglePredictor("results/model/my_own_model.pkl")
        predictor.generate_submission(
            test_csv_path="data/application_test.csv",
            output_csv_path="results/model/kaggle_submission.csv"
        )
    except Exception as main_e:
        logger.critical(f"Prediction script failed: {main_e}")