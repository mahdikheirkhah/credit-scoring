from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    num_impute_strategy: str = "median"
    cat_impute_strategy: str = "constant"
    cat_fill_value: str = "missing"
    # Default switch for our Ablation studies
    use_woe: bool = True
    random_state: int = 42


@dataclass
class LightGBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.085
    max_depth: int = 3
    subsample: float = 0.7
    colsample_bytree: float = 0.8
    class_weight: str = "balanced"
    n_jobs: int = -1
    verbose: int = -1
    random_state: int = 42


@dataclass
class LogisticRegressionConfig:
    class_weight: str = "balanced"
    max_iter: int = 1000
    rfe_c_value: float = 0.1
    rfe_n_features: int = 60
    rfe_step: float = 0.1
    random_state: int = 42


@dataclass
class PipelineConfig:
    model_type: str = "lightgbm"
    random_state: int = 42
    test_size: float = 0.2
    n_cv_splits: int = 5
    shap_sample_size: int = 5000


@dataclass
class GlobalConfig:
    """The master configuration object."""

    # We use field(default_factory=...) to safely instantiate nested classes!
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lgb: LightGBMConfig = field(default_factory=LightGBMConfig)
    lr: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# Instantiate a singleton to be imported across the project
CONFIG = GlobalConfig()
