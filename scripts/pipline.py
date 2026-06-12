import os
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Import our SOLID modules
from feature_engineering import BureauAggregator, PreviousApplicationAggregator, FeatureMerger
from preprocess import BaselinePreprocessor
from feature_selection import RFEFeatureSelector
from models import LogisticRegressionModel, GradientBoostingModel
from predict import CreditRiskPredictor
from config import CONFIG


# ==============================================================================
# PIPELINE MODULES (Single Responsibility Principle)
# ==============================================================================

def load_and_engineer_features(preprocessor: BaselinePreprocessor) -> pd.DataFrame:
    """
    CONTRIBUTING.md Alignment: Data Integrity & Leakage Prevention.
    Aggregates historical data carefully before joining to the main spine to 
    prevent target duplication (leakage).
    """
    logger.info("Loading training and historical datasets...")
    df = preprocessor.load_data("data/application_train.csv")
    bureau_df = preprocessor.load_data("data/bureau.csv")
    prev_app_df = preprocessor.load_data("data/previous_application.csv")
    
    logger.info("Engineering relational proxies via interfaces...")
    bureau_aggregator = BureauAggregator()
    prev_app_aggregator = PreviousApplicationAggregator()
    
    bureau_proxies = bureau_aggregator.aggregate(bureau_df)
    prev_proxies = prev_app_aggregator.aggregate(prev_app_df)
    
    df_merged = FeatureMerger.merge(df, [bureau_proxies, prev_proxies])
    return df_merged


def split_and_preprocess(df: pd.DataFrame, preprocessor: BaselinePreprocessor, use_woe: bool):
    """
    CONTRIBUTING.md Alignment: Class Imbalance Awareness & Reproducibility.
    Uses 'stratify=y' to preserve the rare default ratio. Applies strict fit/transform 
    boundaries to prevent validation leakage. Sets random_state for reproducibility.
    """
    y = df['TARGET']
    X = df.drop(columns=['TARGET']) 

    logger.info("Splitting data with strict stratification...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=CONFIG.pipeline.test_size, 
        random_state=CONFIG.pipeline.random_state, 
        stratify=y
    )
    
    # Isolate IDs for interpretability reports
    train_client_ids = X_train_raw['SK_ID_CURR']
    X_train_raw = X_train_raw.drop(columns=['SK_ID_CURR'])
    X_val_raw = X_val_raw.drop(columns=['SK_ID_CURR'])

    logger.info("Building and applying Preprocessing Pipeline...")
    numeric_cols, categorical_cols = preprocessor.get_feature_types(X_train_raw)
    preprocessor.build_pipeline(numeric_cols, categorical_cols, use_woe=use_woe) 
    
    X_train_processed = preprocessor.fit_transform(X_train_raw, y_train)
    X_val_processed = preprocessor.transform(X_val_raw)
    
    return X_train_processed, X_val_processed, y_train, y_val, train_client_ids

def get_model_name():
    """
    Returns the name of the model based on the configuration.
    """
    return getattr(CONFIG.pipeline, 'model_type', 'lightgbm').lower()

def get_model_from_config():
    """
    OOP Factory Pattern: Returns the correct model instance based on the configuration.
    """
    model_type = get_model_name()
    if model_type == 'logistic':
        return LogisticRegressionModel()
    elif model_type == 'lightgbm':
        return GradientBoostingModel()
    else:
        raise ValueError(f"Unknown model_type in config: {model_type}")


def run_interpretability_reports(model, X_train: pd.DataFrame, client_ids: pd.Series):
    """
    CONTRIBUTING.md Alignment: Interpretability Checks.
    Automatically generates SHAP force plots for tree models, or extracts 
    coefficients for baseline linear models to satisfy regulatory audits.
    """
    logger.info("Generating Interpretability artifacts...")
    if isinstance(model, GradientBoostingModel):
        # Global and Local SHAP for advanced trees
        model.generate_global_shap(X_train.sample(min(len(X_train), CONFIG.pipeline.shap_sample_size), random_state=CONFIG.pipeline.random_state))
        model.generate_local_shap(X_train.iloc[[0]], str(client_ids.iloc[0]), "train")
        model.generate_local_shap(X_train.iloc[[1]], str(client_ids.iloc[1]), "train")
    elif isinstance(model, LogisticRegressionModel):
        # Global Coefficients for baseline linear models
        coef_df = model.extract_coefficients(X_train.columns.tolist())
        os.makedirs("results/model", exist_ok=True)
        coef_df.to_csv("results/model/baseline_coefficients.csv", index=False)
        logger.info(f"Saved baseline coefficients. Top 3 drivers:\n{coef_df.head(3)}")


# ==============================================================================
# ORCHESTRATION (The Maestro)
# ==============================================================================

def run_training_pipeline(use_woe: bool = True, use_rfe: bool = False) -> None:
    try:
        model_name = getattr(CONFIG.pipeline, 'model_type', 'unknown').upper()
        logger.info(f"=== STARTING TRAINING PIPELINE | Model: {model_name} | WoE: {use_woe} | RFE: {use_rfe} ===")
        
        # 1. Initialize Tools
        preprocessor = BaselinePreprocessor()
        model = get_model_from_config()
        
        # 2. Load & Engineer
        df_merged = load_and_engineer_features(preprocessor)
        
        # 3. Split & Preprocess
        X_train, X_val, y_train, y_val, train_ids = split_and_preprocess(df_merged, preprocessor, use_woe)
        
        # 4. Feature Selection (Ablation Switch)
        selector = None
        if use_rfe:
            logger.info("Applying RFE Filter via external selector...")
            rfe_engine = RFEFeatureSelector()
            X_train = rfe_engine.select(X_train, y_train)
            
            # Apply identical filter to validation
            X_val_np = rfe_engine.selector.transform(X_val)
            X_val = pd.DataFrame(X_val_np, columns=rfe_engine.selected_feature_names)
            selector = rfe_engine
        else:
            logger.info("Skipping RFE Filter. Using all features.")

        # 5. Train Model (Polymorphic call)
        model.train(X_train, y_train, X_val, y_val)
        
        # 6. Generate Regulatory Reports
        run_interpretability_reports(model, X_train, train_ids)
        
        # 7. Save Artifacts
        model_name = get_model_name()
        model.save_pipeline(f"results/model/{model_name}_model.pkl", preprocessor, selector)
        logger.info("=== TRAINING PIPELINE COMPLETE ===")
        
    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
        raise


def run_prediction_pipeline() -> None:
    try:
        logger.info("=== STARTING PREDICTION PIPELINE ===")
        
        # 1. Load the unified Predictor
        predictor = CreditRiskPredictor(f"results/model/{get_model_name()}_model.pkl")
        
        # 2. Engineer Test Data exactly like Train Data
        preprocessor = BaselinePreprocessor()
        df_test = preprocessor.load_data("data/application_test.csv")
        bureau_df = preprocessor.load_data("data/bureau.csv")
        prev_app_df = preprocessor.load_data("data/previous_application.csv")
        
        bureau_proxies = BureauAggregator().aggregate(bureau_df)
        prev_proxies = PreviousApplicationAggregator().aggregate(prev_app_df)
        df_test_merged = FeatureMerger.merge(df_test, [bureau_proxies, prev_proxies])

        # 3. Generate Submission
        predictor.generate_kaggle_submission(
            df_test=df_test_merged,
            id_col='SK_ID_CURR',
            output_csv_path="results/model/kaggle_submission.csv"
        )
        logger.info("=== PREDICTION PIPELINE COMPLETE ===")
        
    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("Pipeline Orchestrator Initialized.")
        
        # Ablation Configuration
        run_training_pipeline(use_woe=False, use_rfe=False) 
        run_prediction_pipeline()
        
    except Exception as main_e:
        logger.critical(f"System execution halted due to error: {main_e}")