import pandas as pd
import re
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from category_encoders import WOEEncoder
from scripts.config import CONFIG

class BaselinePreprocessor:
    """Handles data loading, inspection, and stateful preprocessing with zero data leakage."""

    def __init__(self, random_state: int = None) -> None:
        """
        Initializes the preprocessor with a random state and sets up stateful memory.
        
        Args:
            random_state (int, optional): Random seed for reproducibility. Defaults to CONFIG.
            
        Returns:
            None
        """
        try:
            self.random_state = random_state or CONFIG.preprocess.random_state
            self.preprocessor = None
            self.feature_names = None
            # STATEFUL MEMORY: Remembers which columns to drop globally
            self.cols_to_drop = [] 
            logger.info(f"Initialized BaselinePreprocessor with random_state={self.random_state}")
        except Exception as e:
            logger.error(f"Failed to initialize BaselinePreprocessor: {e}")
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads raw data from a CSV file and applies static domain anomaly fixes.
        
        Args:
            filepath (str): The absolute or relative path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded data with static anomaly corrections applied.
        """
        try:
            logger.info(f"Attempting to load data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Domain Anomaly Fix (Static rule for the 1000-year trap)
            if 'DAYS_EMPLOYED' in df.columns:
                df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, float('nan'))
                
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            raise

    def filter_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Dynamically learns which features to drop (Train) or applies memorized drops (Test).
        
        Args:
            X (pd.DataFrame): The input features dataframe.
            fit (bool): If True, learns which columns to drop (redundant/zero-variance). If False, applies memorized columns.
            
        Returns:
            pd.DataFrame: A new dataframe with the problematic columns removed.
        """
        try:
            X_cleaned = X.copy()
            
            if fit:
                logger.info("Dynamically learning features to drop (zero-variance & redundant)...")
                
                # 1. Redundant Building Features
                building_redundant = [c for c in X_cleaned.columns if c.endswith('_AVG') or c.endswith('_MODE')]
                
                # 2. Zero Variance Features (calculated dynamically)
                nunique = X_cleaned.nunique(dropna=True)
                constant_cols = nunique[nunique <= 1].index.tolist()
                
                # Save to stateful memory!
                self.cols_to_drop = list(set(building_redundant + constant_cols))
                logger.info(f"Memorized {len(self.cols_to_drop)} columns to drop globally.")
                
            # Apply the drop using the memory
            existing_drops = [c for c in self.cols_to_drop if c in X_cleaned.columns]
            if existing_drops:
                X_cleaned.drop(columns=existing_drops, inplace=True)
                
            return X_cleaned
        except Exception as e:
            logger.error(f"Failed to filter features: {e}")
            raise

    def get_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Dynamically identifies numerical and categorical feature columns, excluding target and IDs.
        
        Args:
            df (pd.DataFrame): The dataframe to analyze.
            
        Returns:
            tuple[list[str], list[str]]: A tuple containing the list of numerical column names and the list of categorical column names.
        """
        try:
            logger.info("Dynamically separating numerical and categorical features...")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            for col in ['TARGET', 'SK_ID_CURR']:
                if col in numeric_cols: numeric_cols.remove(col)
                if col in categorical_cols: categorical_cols.remove(col)
                
            return numeric_cols, categorical_cols
        except Exception as e:
            logger.error(f"Failed to extract feature types: {e}")
            raise

    def build_pipeline(self, numeric_features: list[str], categorical_features: list[str], use_woe: bool = None) -> None:
        """
        Constructs the Scikit-Learn preprocessing pipeline for numerical and categorical features.
        
        Args:
            numeric_features (list[str]): List of numerical column names to process.
            categorical_features (list[str]): List of categorical column names to process.
            use_woe (bool, optional): Override flag for using Weight of Evidence encoding.
            
        Returns:
            None
        """
        try:
            woe_flag = use_woe if use_woe is not None else CONFIG.preprocess.use_woe

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=CONFIG.preprocess.num_impute_strategy, add_indicator=True)),
                ('scaler', RobustScaler()) 
            ])
            
            if woe_flag:
                logger.info("Using WOEEncoder for categorical features.")
                cat_encoder = WOEEncoder()
            else:
                logger.info("Using OneHotEncoder for categorical features.")
                cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(
                    strategy=CONFIG.preprocess.cat_impute_strategy, 
                    fill_value=CONFIG.preprocess.cat_fill_value,
                    add_indicator=True
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
        """
        Fits the preprocessing pipeline to the training data and transforms it.
        
        Args:
            X (pd.DataFrame): The training input features.
            y (pd.Series, optional): The target variable (required for WoE encoding).
            
        Returns:
            pd.DataFrame: The transformed features as a pandas DataFrame with clean column names.
        """
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
        """
        Transforms test or unseen data using the already fitted pipeline.
        
        Args:
            X (pd.DataFrame): The unseen input features.
            
        Returns:
            pd.DataFrame: The transformed features as a pandas DataFrame with clean column names.
        """
        try:
            if self.preprocessor is None or self.feature_names is None:
                raise ValueError("The pipeline has not been fitted yet.")
            
            X_processed = self.preprocessor.transform(X)
            return pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)
        except Exception as e:
            logger.error(f"Failed during transform: {e}")
            raise