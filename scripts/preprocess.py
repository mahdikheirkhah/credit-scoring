import pandas as pd
import re
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import WOEEncoder

# Inject the centralized configuration
from scripts.config import CONFIG


class BaselinePreprocessor:
    """Handles data loading, inspection, and preprocessing with zero data leakage."""

    def __init__(self, random_state: int = None) -> None:
        try:
            # Fall back to config if no specific state is passed
            self.random_state = random_state or CONFIG.preprocess.random_state
            self.preprocessor = None
            self.feature_names = None
            logger.info(f"Initialized BaselinePreprocessor with random_state={self.random_state}")
        except Exception as e:
            logger.error(f"Failed to initialize BaselinePreprocessor: {e}")
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            logger.info(f"Attempting to load data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found at {filepath}. Please ensure it exists.")
            raise

    def log_column_types(self, df: pd.DataFrame) -> None:
        try:
            logger.info("Inspecting column data types...")
            dtypes_str = "\n" + df.dtypes.to_string()
            logger.info(f"Dataset Column Types: {dtypes_str}")
        except Exception as e:
            logger.error(f"Failed to log column types: {e}")
            raise

    def get_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        try:
            logger.info("Dynamically separating numerical and categorical features...")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            logger.info(f"Identified {len(numeric_cols)} numerical and {len(categorical_cols)} categorical features.")
            return numeric_cols, categorical_cols
        except Exception as e:
            logger.error(f"Failed to extract feature types: {e}")
            raise

    def build_pipeline(self, numeric_features: list[str], categorical_features: list[str], use_woe: bool = None) -> None:
        """Builds the preprocessing pipeline with a switchable categorical encoder."""
        try:
            # Determine ablation configuration dynamically
            woe_flag = use_woe if use_woe is not None else CONFIG.preprocess.use_woe

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=CONFIG.preprocess.num_impute_strategy)),
                ('scaler', StandardScaler())
            ])
            
            # --- THE SWITCH LOGIC ---
            if woe_flag:
                logger.info("Using WOEEncoder for categorical features.")
                cat_encoder = WOEEncoder()
            else:
                logger.info("Using OneHotEncoder for categorical features (WoE disabled).")
                cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(
                    strategy=CONFIG.preprocess.cat_impute_strategy, 
                    fill_value=CONFIG.preprocess.cat_fill_value
                )),
                ('encoder', cat_encoder) 
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            logger.info("Successfully built preprocessing pipeline.")
        except Exception as e:
            logger.error(f"Failed to build preprocessing pipeline: {e}")
            raise

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        try:
            logger.info("Fitting and transforming data...")
            X_processed = self.preprocessor.fit_transform(X, y)
            
            raw_names = self.preprocessor.get_feature_names_out()
            self.feature_names = [re.sub(r'[\[\]{},:" ]', '_', name) for name in raw_names]
            
            logger.info(f"Successfully processed data. New shape: {X_processed.shape}")
            return pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)
        except Exception as e:
            logger.error(f"Failed during fit_transform: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Transforming validation/test data...")
            if self.preprocessor is None or self.feature_names is None:
                raise ValueError("The pipeline has not been fitted yet. Call fit_transform first.")
            
            X_processed = self.preprocessor.transform(X)
            logger.info(f"Successfully transformed data. Shape: {X_processed.shape}")
            return pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)
        except Exception as e:
            logger.error(f"Failed during transform: {e}")
            raise