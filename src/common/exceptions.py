"""Custom exceptions for the MedicalXAI system."""


class MedicalAIError(Exception):
    """Base exception for all MedicalXAI errors."""


class DataLoadError(MedicalAIError):
    """Raised when a dataset or file cannot be loaded."""


class ConfigError(MedicalAIError):
    """Raised when configuration is missing or invalid."""


class ModelError(MedicalAIError):
    """Raised on model init or forward-pass failure."""


class CheckpointError(MedicalAIError):
    """Raised on checkpoint save/load failure."""


class BundleValidationError(MedicalAIError):
    """Raised when MONAI bundle structure validation fails."""


class MetricComputationError(MedicalAIError):
    """Raised when metric computation encounters an error."""


class LabelMappingError(MedicalAIError):
    """Raised when label encoding/decoding fails."""
