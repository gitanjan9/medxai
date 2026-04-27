"""Pydantic schemas for data validation and label encoding."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator


class DataSample(BaseModel):
    """Schema for a single processed data sample."""

    label: int
    label_str: str
    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    findings: Optional[str] = None
    patient_id: Optional[str] = None

    @field_validator("label")
    @classmethod
    def label_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("label must be >= 0")
        return v


class LabelMap(BaseModel):
    """Bidirectional label encoder/decoder for string class names."""

    str_to_idx: dict[str, int]
    idx_to_str: dict[str, str]   # JSON keys are always strings
    num_classes: int

    @classmethod
    def from_labels(cls, labels: list[str]) -> "LabelMap":
        """Build a LabelMap from a list of raw label strings."""
        sorted_labels = sorted(set(labels))
        str_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}
        idx_to_str = {str(i): lbl for lbl, i in str_to_idx.items()}
        return cls(
            str_to_idx=str_to_idx,
            idx_to_str=idx_to_str,
            num_classes=len(sorted_labels),
        )

    def encode(self, label_str: str) -> int:
        """Convert a label string to its integer index."""
        if label_str not in self.str_to_idx:
            raise KeyError(f"Unknown label: '{label_str}'")
        return self.str_to_idx[label_str]

    def decode(self, idx: int) -> str:
        """Convert an integer index to its label string."""
        key = str(idx)
        if key not in self.idx_to_str:
            raise KeyError(f"Unknown index: {idx}")
        return self.idx_to_str[key]

    def class_names(self) -> list[str]:
        """Return class names in index order."""
        return [self.idx_to_str[str(i)] for i in range(self.num_classes)]

    def save(self, path: Path) -> None:
        """Persist to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.model_dump(), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LabelMap":
        """Load from a JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)
