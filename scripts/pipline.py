import os
import pandas as pd
from loguru import logger

from feature_engineering import RelationalFeatureEngineer
from preprocess import BaselinePreprocessor
from train import EconometricBaselineModel 
from predict import KagglePredictor

def run_training_pipeline() -> None:
    try:
        logger.info("=== STARTING TRAINING PIPELINE ===")
        feature_engineer = RelationalFeatureEngineer()
        preprocessor = BaselinePreprocessor()
        baseline_model = EconometricBaselineModel()

        logger.info("Loading training and historical datasets...")
        df = preprocessor.load_data("data/application_train.csv")
        bureau_df = preprocessor.load_data("data/bureau.csv")
        prev_app_df = preprocessor.load_data("data/previous_application.csv")
        
        bureau_proxies = feature_engineer.aggregate_bureau(bureau_df)
        prev_proxies = feature_engineer.aggregate_previous_applications(prev_app_df)
        df = feature_engineer.merge_features(df, [bureau_proxies, prev_proxies])
        
        y = df['TARGET']
        X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

        numeric_cols, categorical_cols = preprocessor.get_feature_types(X)
        preprocessor.build_pipeline(numeric_cols, categorical_cols)
        X_processed = preprocessor.fit_transform(X, y) 

        X_selected = baseline_model.select_features(X_processed, y, n_features=30)
        baseline_model.evaluate_cv(X_selected, y)
        
        logger.info("Fitting final global model...")
        baseline_model.model.fit(X_selected, y)
        
        coef_df = baseline_model.extract_coefficients(baseline_model.selected_feature_names)
        baseline_model.save_pipeline(preprocessor, "results/model/my_own_model.pkl")
        
        os.makedirs("results/model", exist_ok=True)
        coef_df.to_csv("results/model/baseline_coefficients.csv", index=False)
        logger.info(f"Top 5 strongest predictors:\n{coef_df.head(5)}")
        logger.info("=== TRAINING PIPELINE COMPLETE ===")
    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
        raise

def run_prediction_pipeline() -> None:
    try:
        logger.info("=== STARTING PREDICTION PIPELINE ===")
        feature_engineer = RelationalFeatureEngineer()
        predictor = KagglePredictor("results/model/my_own_model.pkl")
        
        logger.info("Loading test and historical datasets...")
        df_test_raw = pd.read_csv("data/application_test.csv")
        bureau_df = pd.read_csv("data/bureau.csv")
        prev_app_df = pd.read_csv("data/previous_application.csv")
        
        bureau_proxies = feature_engineer.aggregate_bureau(bureau_df)
        prev_proxies = feature_engineer.aggregate_previous_applications(prev_app_df)
        df_test_merged = feature_engineer.merge_features(df_test_raw, [bureau_proxies, prev_proxies])

        predictor.generate_submission(
            df_test=df_test_merged,
            output_csv_path="results/model/kaggle_submission.csv"
        )
        logger.info("=== PREDICTION PIPELINE COMPLETE ===")
    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Pipeline Orchestrator Initialized.")
        run_training_pipeline()
        run_prediction_pipeline()
    except Exception as main_e:
        logger.critical(f"System execution halted due to error: {main_e}")