"""Pydantic v2 configuration models for the MedicalXAI training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    name: str = "mimic_cxr_classifier"
    seed: int = 42
    output_dir: Path = Path("artifacts")


class DataConfig(BaseModel):
    mode: Literal["image", "text", "multimodal"] = "image"
    task: Literal["multiclass", "multilabel"] = "multiclass"
    train_path: Path = Path("data/train.csv")
    val_path: Optional[Path] = None
    test_path: Optional[Path] = None
    val_split: float = Field(0.15, ge=0.0, lt=1.0)
    image_col: str = "image"
    label_col: str = "impression"
    label_cols: Optional[list[str]] = None  # for multilabel: list of binary label columns
    text_col: Optional[str] = "findings"
    patient_id_col: Optional[str] = None
    image_size: list[int] = Field(default_factory=lambda: [224, 224])
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True
    label_mapping_path: Optional[Path] = None
    label_map_csv: Optional[Path] = None  # CSV used to derive the 17-class label map
    class_merge_map_path: Optional[Path] = None  # JSON merge map to collapse near-identical classes


class ModelConfig(BaseModel):
    architecture: Literal["densenet121", "efficientnet_b0", "efficientnet_b3", "efficientnet_b4", "resnet50", "resnet18"] = "densenet121"
    spatial_dims: int = 2
    in_channels: int = 1
    num_classes: int = 17
    pretrained: bool = True
    dropout_rate: float = Field(0.3, ge=0.0, le=0.9)
    freeze_backbone: bool = False


class TrainingConfig(BaseModel):
    epochs: int = Field(50, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(1e-4, gt=0.0)
    weight_decay: float = Field(1e-5, ge=0.0)
    use_amp: bool = True
    grad_clip_norm: float = Field(1.0, gt=0.0)
    class_balance_strategy: Literal["weighted_loss", "weighted_sampler", "focal", "class_balanced_focal", "multilabel_bce_focal"] = "weighted_loss"
    freeze_epochs: int = Field(0, ge=0)  # epochs to train head-only before unfreezing backbone
    focal_gamma: float = Field(2.0, ge=0.0)  # focusing parameter for focal / class_balanced_focal
    backbone_lr_scale: float = Field(0.1, gt=0.0)  # backbone LR = learning_rate * backbone_lr_scale


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = Field(7, ge=1)
    monitor: str = "val_auroc_macro"
    mode: Literal["min", "max"] = "max"
    min_delta: float = Field(1e-4, ge=0.0)


class OptimizerConfig(BaseModel):
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = Field(1e-8, gt=0.0)
    momentum: float = Field(0.9, ge=0.0, le=1.0)


class SchedulerConfig(BaseModel):
    name: Literal[
        "cosine_annealing",
        "cosine_annealing_warm_restarts",
        "step",
        "none",
    ] = "cosine_annealing_warm_restarts"
    T_0: int = 10
    T_mult: int = 2
    T_max: int = 50
    eta_min: float = 1e-7
    step_size: int = 10
    gamma: float = 0.1


class CheckpointConfig(BaseModel):
    dir: Path = Path("artifacts/checkpoints")
    save_top_k: int = Field(3, ge=1)
    monitor: str = "val_auroc_macro"
    mode: Literal["min", "max"] = "max"
    filename_template: str = "epoch={epoch:03d}_auroc={val_auroc_macro:.4f}"


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "mimic_cxr_day1"
    run_name: Optional[str] = None
    log_model_artifact: bool = True
    artifact_path: str = "model"


class BundleConfig(BaseModel):
    name: str = "mimic_cxr_classifier"
    bundle_root: Path = Path("bundles/classifier_bundle")
    version: str = "0.1.0"


class CalibrationConfig(BaseModel):
    method: Literal["temperature_scaling"] = "temperature_scaling"
    max_iter: int = Field(50, ge=1)
    lr: float = Field(0.01, gt=0.0)
    output_path: Path = Path("artifacts/calibration.json")


class ThresholdConfig(BaseModel):
    method: Literal["ppv_recall_balanced"] = "ppv_recall_balanced"
    target_ppv: float = Field(0.65, ge=0.0, le=1.0)
    target_recall: float = Field(0.70, ge=0.0, le=1.0)
    default_low: float = Field(0.25, ge=0.0, le=1.0)
    default_high: float = Field(0.65, ge=0.0, le=1.0)
    output_path: Path = Path("artifacts/thresholds.json")
    review_band_enabled: bool = True


class ExplainabilityConfig(BaseModel):
    method: Literal["gradcam"] = "gradcam"
    target_layer: str = "features.denseblock4"
    output_dir: Path = Path("artifacts/explanations")
    save_heatmaps: bool = True
    max_samples: Optional[int] = None  # None = all val samples


class ReviewConfig(BaseModel):
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    require_explanation: bool = True
    flag_weak_calibration: bool = True


# ---------------------------------------------------------------------------
# Inference root config (Day 2+, consumed by serving / Day 3)
# ---------------------------------------------------------------------------


class InferenceConfig(BaseModel):
    """Standalone config for inference, calibration, and explainability.
    Loaded from configs/inference.yaml."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    checkpoint_path: Optional[Path] = None
    label_map_path: Path = Path("artifacts/label_map.json")
    calibration_path: Optional[Path] = Path("artifacts/calibration.json")
    thresholds_path: Optional[Path] = Path("artifacts/thresholds.json")
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "InferenceConfig":
        import yaml
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class TrainConfig(BaseModel):
    """Root configuration model – loaded from train.yaml."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load and validate config from a YAML file."""
        import yaml  # local import to keep this module lightweight

        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)
