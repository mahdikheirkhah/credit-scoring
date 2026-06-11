import os
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from feature_engineering import RelationalFeatureEngineer
from preprocess import BaselinePreprocessor
from predict import KagglePredictor
from advanced_model import GradientBoostingModel  # <-- Using our LightGBM Engine

def run_training_pipeline() -> None:
    try:
        logger.info("=== STARTING ADVANCED TRAINING PIPELINE ===")
        feature_engineer = RelationalFeatureEngineer()
        preprocessor = BaselinePreprocessor()
        advanced_model = GradientBoostingModel()

        logger.info("Loading training and historical datasets...")
        df = preprocessor.load_data("data/application_train.csv")
        bureau_df = preprocessor.load_data("data/bureau.csv")
        prev_app_df = preprocessor.load_data("data/previous_application.csv")
        
        # 1. Use your EXPLICIT aggregation methods
        logger.info("Engineering relational proxies...")
        bureau_proxies = feature_engineer.aggregate_bureau(bureau_df)
        prev_proxies = feature_engineer.aggregate_previous_applications(prev_app_df)
        df = feature_engineer.merge_features(df, [bureau_proxies, prev_proxies])
        
        y = df['TARGET']
        X = df.drop(columns=['TARGET']) 

        # 2. Split the data BEFORE preprocessing to guarantee zero leakage and to enable Learning Curves
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save the client IDs for our SHAP reports, then drop them
        train_client_ids = X_train_raw['SK_ID_CURR']
        X_train_raw = X_train_raw.drop(columns=['SK_ID_CURR'])
        X_val_raw = X_val_raw.drop(columns=['SK_ID_CURR'])

        # 3. Preprocessing & WoE Encoding
        numeric_cols, categorical_cols = preprocessor.get_feature_types(X_train_raw)
        preprocessor.build_pipeline(numeric_cols, categorical_cols)
        
        X_train_processed = preprocessor.fit_transform(X_train_raw, y_train)
        X_val_processed = preprocessor.transform(X_val_raw) # STRICTLY transform() for validation
        
        # 4. Train LightGBM & Generate Learning Curves
        advanced_model.train_with_learning_curves(X_train_processed, y_train, X_val_processed, y_val)
        
        # 5. Interpretability: Global & Local SHAP (2 Train Clients)
        # We take a sample of 5000 rows for the global plot so it generates quickly
        advanced_model.generate_global_shap(X_train_processed.sample(5000, random_state=42))
        advanced_model.generate_local_shap(X_train_processed.iloc[[0]], str(train_client_ids.iloc[0]), "train")
        advanced_model.generate_local_shap(X_train_processed.iloc[[1]], str(train_client_ids.iloc[1]), "train")
        
        # 6. Save Pipeline
        advanced_model.save_pipeline(preprocessor, "results/model/my_own_model.pkl")
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
        
        # 1. Use your EXPLICIT aggregation methods
        bureau_proxies = feature_engineer.aggregate_bureau(bureau_df)
        prev_proxies = feature_engineer.aggregate_previous_applications(prev_app_df)
        df_test_merged = feature_engineer.merge_features(df_test_raw, [bureau_proxies, prev_proxies])

        # 2. Generate 1 Local SHAP plot for a test client before prediction
        logger.info("Generating SHAP report for 1 test client...")
        test_client_id = df_test_merged['SK_ID_CURR'].iloc[0]
        test_client_raw = df_test_merged.drop(columns=['SK_ID_CURR']).iloc[[0]]
        test_client_processed = predictor.preprocessor.transform(test_client_raw)
        test_client_df = pd.DataFrame(test_client_processed, columns=predictor.feature_names)
        
        import shap
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(test_client_df)
        if isinstance(shap_values, list): shap_values = shap_values[1]
        
        force_plot = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
            shap_values[0, :], 
            test_client_df.iloc[0, :]
        )
        shap.save_html(f"results/clients_outputs/test_client_{test_client_id}_force_plot.html", force_plot)

        # 3. Generate standard Kaggle Submission
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