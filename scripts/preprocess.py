import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BaselinePreprocessor:
    """
    Handles data loading, inspection, and preprocessing for the baseline credit scoring model.
    Ensures zero data leakage by strictly separating fit and transform phases.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes the preprocessor with a fixed random state for reproducibility.
        
        Args:
            random_state (int): Seed for reproducibility.
        """
        try:
            self.random_state = random_state
            self.preprocessor = None
            self.feature_names = None
            logger.info(f"Initialized BaselinePreprocessor with random_state={self.random_state}")
        except Exception as e:
            logger.error(f"Failed to initialize BaselinePreprocessor: {e}")
            raise

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads the application data from a CSV file.
        
        Args:
            filepath (str): The path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            logger.info(f"Attempting to load data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found at {filepath}. Please ensure it exists.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading data: {e}")
            raise

    def log_column_types(self, df: pd.DataFrame) -> None:
        """
        Logs the data types of all columns in the provided DataFrame.
        Replaces standard printing to maintain a clean audit trail.
        
        Args:
            df (pd.DataFrame): The dataset to inspect.
        """
        try:
            logger.info("Inspecting column data types...")
            # We convert the dtypes Series to a string to log it cleanly
            dtypes_str = "\n" + df.dtypes.to_string()
            logger.info(f"Dataset Column Types: {dtypes_str}")
        except Exception as e:
            logger.error(f"Failed to log column types: {e}")
            raise

    def get_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Dynamically identifies and separates numerical and categorical columns.
        
        Args:
            df (pd.DataFrame): The dataset to analyze (excluding target and ID columns).
            
        Returns:
            tuple[list[str], list[str]]: A tuple containing the list of numerical 
                                         feature names and categorical feature names.
        """
        try:
            logger.info("Dynamically separating numerical and categorical features...")
            
            # Extract numerical columns (integers and floats)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Extract categorical columns (objects, categories, and booleans)
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            logger.info(f"Identified {len(numeric_cols)} numerical and {len(categorical_cols)} categorical features.")
            return numeric_cols, categorical_cols
            
        except Exception as e:
            logger.error(f"Failed to extract feature types: {e}")
            raise

    def build_pipeline(self, numeric_features: list[str], categorical_features: list[str]) -> None:
        """
        Builds the Scikit-Learn preprocessing pipeline to handle missing values and scaling.
        
        Args:
            numeric_features (list[str]): List of numeric column names.
            categorical_features (list[str]): List of categorical column names.
        """
        try:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the pipeline on the training data and transforms it.
        
        Args:
            X (pd.DataFrame): The raw training features.
            
        Returns:
            pd.DataFrame: The preprocessed training features.
        """
        try:
            logger.info("Fitting and transforming data...")
            X_processed = self.preprocessor.fit_transform(X)
            self.feature_names = self.preprocessor.get_feature_names_out()
            
            logger.info(f"Successfully processed data. New shape: {X_processed.shape}")
            return pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)
        except Exception as e:
            logger.error(f"Failed during fit_transform: {e}")
            raise