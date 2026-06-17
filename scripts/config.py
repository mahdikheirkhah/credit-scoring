from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    num_impute_strategy: str = "median"
    cat_impute_strategy: str = "constant"
    cat_fill_value: str = "missing"
    # Default switch for our Ablation studies
    use_woe: bool = False
    random_state: int = 42


@dataclass
class LightGBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.085
    max_depth: int = 3
    subsample: float = 0.5
    colsample_bytree: float = 0.8
    class_weight: str = "balanced"
    # class_weight: dict = field(default_factory=lambda: {0: 1.0, 1: 128.0})
    n_jobs: int = -1
    verbose: int = -1
    random_state: int = 42
    early_stopping_rounds: int = 50
    linear_tree: bool = False

@dataclass
class LogisticRegressionConfig:
    class_weight: str = "balanced"
    max_iter: int = 100000
    rfe_c_value: float = 0.1
    rfe_n_features: int = 250
    rfe_step: float = 0.1
    rfe_max_iter: int = 100000
    random_state: int = 42


@dataclass
class PiecewiseConfig:
    # Tree Parameters
    max_depth: int = 3  # Creates up to 8 distinct "Borrower Personas"
    min_samples_leaf: int = 270000  # Ensures enough data in each leaf for stable LR
    tree_class_weight: str = "balanced"
    
    # Logistic Regression Parameters
    lr_c_value: float = 0.1 # Strong regularization
    lr_class_weight: str = "balanced"
    random_state: int = 42
    lr_max_iter: int = 100000

@dataclass
class PipelineConfig:
    model_type: str = "lightgbm"
    random_state: int = 42
    test_size: float = 0.15
    n_cv_splits: int = 5
    shap_sample_size: int = 10000
    use_rfe: bool = False

@dataclass
class GlobalConfig:
    """The master configuration object."""

    # We use field(default_factory=...) to safely instantiate nested classes!
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lgb: LightGBMConfig = field(default_factory=LightGBMConfig)
    lr: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    piecewise: PiecewiseConfig = field(default_factory=PiecewiseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Instantiate a singleton to be imported across the project
CONFIG = GlobalConfig()
