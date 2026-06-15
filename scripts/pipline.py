import os
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# Import our SOLID modules
from scripts.feature_engineering import (
    BureauAggregator,
    PreviousApplicationAggregator,
    FeatureMerger,
)
from scripts.preprocess import BaselinePreprocessor
from scripts.feature_selection import RFEFeatureSelector
from scripts.models import LogisticRegressionModel, GradientBoostingModel, PiecewiseLinearModel
from scripts.predict import CreditRiskPredictor
from scripts.config import CONFIG

# ==============================================================================
# PIPELINE MODULES (Single Responsibility Principle)
# ==============================================================================


def load_and_engineer_features(preprocessor: BaselinePreprocessor, dataset_type: str = "train") -> pd.DataFrame:
    """Loads all 7 tables, applies Two-Step Aggregation, and merges onto the correct spine (train or test)."""
    
    logger.info(f"Loading {dataset_type} and historical datasets...")
    
    # Dynamically select the spine based on the pipeline phase
    spine_file = "data/application_train.csv" if dataset_type == "train" else "data/application_test.csv"
    df = preprocessor.load_data(spine_file)
    
    bureau_df = preprocessor.load_data("data/bureau.csv")
    bureau_bal_df = preprocessor.load_data("data/bureau_balance.csv")
    prev_app_df = preprocessor.load_data("data/previous_application.csv")
    pos_cash_df = preprocessor.load_data("data/POS_CASH_balance.csv")
    credit_card_df = preprocessor.load_data("data/credit_card_balance.csv")
    installments_df = preprocessor.load_data("data/installments_payments.csv")
    
    logger.info("Engineering relational proxies via interfaces...")
    
    from scripts.feature_engineering import (
        BureauBalanceAggregator, BureauAggregator,
        PreviousApplicationAggregator, POSCashAggregator,
        CreditCardAggregator, InstallmentsAggregator, FeatureMerger
    )
    
    # 1. External Bureau (1:N:M -> Merge -> Aggregate)
    bb_proxies = BureauBalanceAggregator.aggregate(bureau_bal_df)
    bureau_enriched = bureau_df.merge(bb_proxies, on='SK_ID_BUREAU', how='left')
    bureau_proxies = BureauAggregator().aggregate(bureau_enriched)
    
    # 2. Internal Previous Applications (1:N)
    prev_proxies = PreviousApplicationAggregator().aggregate(prev_app_df)
    
    # 3. Internal POS Cash (1:N)
    pos_proxies = POSCashAggregator().aggregate(pos_cash_df)
    
    # 4. Internal Credit Cards (1:N:M -> Two-Step Aggregation)
    cc_proxies = CreditCardAggregator().aggregate(credit_card_df)
    
    # 5. Internal Installments (1:N:M -> Two-Step Aggregation)
    inst_proxies = InstallmentsAggregator().aggregate(installments_df)
    
    # Merge all historical features onto the main spine
    feature_dfs = [bureau_proxies, prev_proxies, pos_proxies, cc_proxies, inst_proxies]
    df_merged = FeatureMerger.merge(df, feature_dfs)
    
    return df_merged

# UPDATE THIS FUNCTION in pipeline.py
def split_and_preprocess(
    df: pd.DataFrame, preprocessor: BaselinePreprocessor, use_woe: bool
):
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    logger.info("Splitting data with strict stratification...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=CONFIG.pipeline.test_size,
        random_state=CONFIG.pipeline.random_state,
        stratify=y,
    )

    train_client_ids = X_train_raw["SK_ID_CURR"]
    X_train_raw = X_train_raw.drop(columns=["SK_ID_CURR"])
    X_val_raw = X_val_raw.drop(columns=["SK_ID_CURR"])

    # ---> NEW: DYNAMIC STATEFUL FILTERING <---
    X_train_raw = preprocessor.filter_features(
        X_train_raw, fit=True
    )  # Learns and Drops
    X_val_raw = preprocessor.filter_features(X_val_raw, fit=False)  # Only Drops

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
    return getattr(CONFIG.pipeline, "model_type", "lightgbm").lower()


def get_model_from_config(override_type=None):
    model_type = override_type or getattr(CONFIG.pipeline, "model_type", "lightgbm").lower()
    
    if model_type == "logistic":
        return LogisticRegressionModel()
    elif model_type == "lightgbm":
        return GradientBoostingModel()
    elif model_type == "piecewise":
        return PiecewiseLinearModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_interpretability_reports(model, X_train: pd.DataFrame, client_ids: pd.Series):
    """
    CONTRIBUTING.md Alignment: Interpretability Checks.
    Automatically generates SHAP force plots for tree models, or extracts
    coefficients for baseline linear models to satisfy regulatory audits.
    """
    logger.info("Generating Interpretability artifacts...")
    if isinstance(model, GradientBoostingModel):
        # Global and Local SHAP for advanced trees
        model.generate_global_shap(
            X_train.sample(
                min(len(X_train), CONFIG.pipeline.shap_sample_size),
                random_state=CONFIG.pipeline.random_state,
            )
        )
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
        model_name = getattr(CONFIG.pipeline, "model_type", "unknown").upper()
        logger.info(f"=== STARTING TRAINING PIPELINE | Model: {model_name} | WoE: {use_woe} | RFE: {use_rfe} ===")

        preprocessor = BaselinePreprocessor()
        model = get_model_from_config()

        # Generate features for the TRAIN spine
        df_merged = load_and_engineer_features(preprocessor, dataset_type="train")

        X_train, X_val, y_train, y_val, train_ids = split_and_preprocess(df_merged, preprocessor, use_woe)

        selector = None
        if use_rfe:
            logger.info("Applying RFE Filter via external selector...")
            rfe_engine = RFEFeatureSelector()
            X_train = rfe_engine.select(X_train, y_train)

            X_val_np = rfe_engine.selector.transform(X_val)
            X_val = pd.DataFrame(X_val_np, columns=rfe_engine.selected_feature_names)
            selector = rfe_engine
        else:
            logger.info("Skipping RFE Filter. Using all features.")

        model.train(X_train, y_train, X_val, y_val)
        run_interpretability_reports(model, X_train, train_ids)
        
        model.save_pipeline(f"results/model/{get_model_name()}_model.pkl", preprocessor, selector)
        logger.info("=== TRAINING PIPELINE COMPLETE ===")

    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
        raise


def run_prediction_pipeline() -> None:
    try:
        logger.info("=== STARTING PREDICTION PIPELINE ===")

        predictor = CreditRiskPredictor(f"results/model/{get_model_name()}_model.pkl")
        preprocessor = BaselinePreprocessor()

        # Generate exact same features, but for tßhe TEST spine!
        df_test_merged = load_and_engineer_features(preprocessor, dataset_type="test")

        predictor.generate_kaggle_submission(
            df_test=df_test_merged,
            id_col="SK_ID_CURR",
            output_csv_path="results/model/kaggle_submission.csv",
        )
        logger.info("=== PREDICTION PIPELINE COMPLETE ===")

    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}")
        raise

def run_model_comparison():
    """Runs an A/B/C test across the three architectures."""
    logger.info("=== STARTING ARCHITECTURE COMPARISON ===")
    
    preprocessor = BaselinePreprocessor()
    df_merged = load_and_engineer_features(preprocessor, dataset_type="train")
    
    # We must use WOE for the linear models to handle categorical variables!
    X_train, X_val, y_train, y_val, _ = split_and_preprocess(df_merged, preprocessor, use_woe=True)
    
    architectures = ["logistic", "piecewise", "lightgbm"]
    results = {}
    
    
    for arch in architectures:
        model = get_model_from_config(override_type=arch)
        model.train(X_train, y_train, X_val, y_val)
        
        preds = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, preds)
        results[arch] = auc
        logger.info(f"Architecture [{arch.upper()}] Validation AUC: {auc:.4f}")
        
    logger.info(f"Comparison Complete: {results}")

if __name__ == "__main__":
    try:
        logger.info("Pipeline Orchestrator Initialized.")
        #run_model_comparison()
        # Ablation Configuration
        run_training_pipeline(use_woe=CONFIG.preprocess.use_woe, use_rfe=CONFIG.pipeline.use_rfe)
        run_prediction_pipeline()

    except Exception as main_e:
        logger.critical(f"System execution halted due to error: {main_e}")
