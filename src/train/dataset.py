"""Dataset implementation supporting inline image bytes and file-path images."""
from __future__ import annotations

import ast
import io
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.common.exceptions import DataLoadError, LabelMappingError
from src.common.schemas import LabelMap

logger = logging.getLogger("medicalxai.dataset")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CXRDataset(Dataset):
    """Chest X-ray classification dataset.

    Supports two image sources (auto-detected per row):
    - **inline bytes**: ``image`` column contains a Python bytes literal
      string (e.g. ``b'\\xff\\xd8...'``).
    - **file path**: ``image_path`` column (or same column) contains a
      resolvable path string.

    Args:
        df: Pre-loaded DataFrame with at least ``image_col`` and
            ``label_col`` columns.
        label_map: Fitted :class:`~src.common.schemas.LabelMap`.
        image_col: Column name holding image bytes or file paths.
        label_col: Column name holding the class label string.
        text_col: Optional column name for auxiliary findings text.
        transforms: Callable applied to the decoded ``np.ndarray`` image
            (C×H×W, float32, in [0, 1]).
        cache_images: Pre-decode all images into RAM.  Useful for small
            datasets on fast storage; leave False for large datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_map: LabelMap,
        image_col: str = "image",
        label_col: str = "impression",
        text_col: Optional[str] = "findings",
        transforms: Optional[Callable] = None,
        cache_images: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.image_col = image_col
        self.label_col = label_col
        self.text_col = text_col
        self.transforms = transforms
        self._cache: dict[int, np.ndarray] = {}

        self._has_images = image_col in df.columns
        if not self._has_images:
            logger.warning(
                "Column '%s' not found – dataset will return zero images.", image_col
            )

        if cache_images and self._has_images:
            logger.info("Pre-caching %d images …", len(self.df))
            for i in range(len(self.df)):
                self._cache[i] = self._load_image(i)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # ---- Label ----
        label_str = str(row[self.label_col])
        try:
            label_idx = self.label_map.encode(label_str)
        except KeyError as exc:
            raise LabelMappingError(str(exc)) from exc

        # ---- Image ----
        if self._has_images:
            img_arr = self._cache.get(idx) or self._load_image(idx)
            if self.transforms is not None:
                img_arr = self.transforms(img_arr)
            image_tensor = torch.as_tensor(img_arr, dtype=torch.float32)
        else:
            image_tensor = torch.zeros(1, 1, 1, dtype=torch.float32)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_str": label_str,
        }

        if self.text_col and self.text_col in self.df.columns:
            sample["findings"] = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""

        return sample

    def class_counts(self) -> dict[int, int]:
        """Return {class_idx: count} for the dataset split."""
        counts: dict[int, int] = {}
        for lbl in self.df[self.label_col]:
            idx = self.label_map.encode(str(lbl))
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Return inverse-frequency class weights as a float32 tensor."""
        from sklearn.utils.class_weight import compute_class_weight

        labels_int = [
            self.label_map.encode(str(l)) for l in self.df[self.label_col]
        ]
        classes = np.arange(self.label_map.num_classes)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=np.array(labels_int)
        )
        return torch.tensor(weights, dtype=torch.float32)

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Build a :class:`WeightedRandomSampler` for balanced mini-batches."""
        class_w = self.compute_class_weights().numpy()
        sample_w = [
            class_w[self.label_map.encode(str(l))]
            for l in self.df[self.label_col]
        ]
        return WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float64),
            num_samples=len(sample_w),
            replacement=True,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_image(self, idx: int) -> np.ndarray:
        """Decode one image row → float32 ndarray (1×H×W, [0,1])."""
        raw_val = self.df.iloc[idx][self.image_col]
        return _decode_image_value(raw_val)


# ---------------------------------------------------------------------------
# Image value decoder
# ---------------------------------------------------------------------------


def _decode_image_value(raw: Any) -> np.ndarray:
    """Decode an image column value into a float32 CHW numpy array.

    Handles:
    - Python bytes literal strings like ``b'\\xff\\xd8...'``
    - Raw :class:`bytes` objects
    - File path strings / :class:`Path` objects
    """
    if isinstance(raw, bytes):
        img_bytes = raw
    elif isinstance(raw, str) and raw.startswith("b'"):
        try:
            img_bytes = ast.literal_eval(raw)
        except Exception as exc:
            raise DataLoadError(f"Cannot parse bytes literal: {exc}") from exc
    elif isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.exists():
            raise DataLoadError(f"Image file not found: {path}")
        with open(path, "rb") as fh:
            img_bytes = fh.read()
    else:
        raise DataLoadError(f"Unsupported image column type: {type(raw)}")

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    except Exception as exc:
        raise DataLoadError(f"PIL failed to open image: {exc}") from exc

    arr = np.array(pil_img, dtype=np.float32) / 255.0    # H×W, [0,1]
    arr = np.expand_dims(arr, axis=0)                     # 1×H×W
    return arr


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_datasets(
    train_path: Union[str, Path],
    label_map: LabelMap,
    val_path: Optional[Union[str, Path]] = None,
    val_split: float = 0.15,
    image_col: str = "image",
    label_col: str = "impression",
    text_col: Optional[str] = "findings",
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    seed: int = 42,
    merge_map_path: Optional[Union[str, Path]] = None,
) -> tuple[CXRDataset, CXRDataset]:
    """Load CSVs and return ``(train_ds, val_ds)``.

    If *val_path* is ``None``, a stratified *val_split* fraction of the
    training CSV is held out as validation.
    If *merge_map_path* is provided, class labels are remapped before filtering.
    """
    from sklearn.model_selection import train_test_split

    train_df = _read_csv(train_path)
    if merge_map_path and Path(merge_map_path).exists():
        train_df = apply_class_merge(train_df, label_col, merge_map_path)
    train_df = _filter_known_labels(train_df, label_col, label_map)
    logger.info("Loaded train CSV: %s rows from %s", len(train_df), train_path)

    if val_path is not None:
        val_df = _read_csv(val_path)
        if merge_map_path and Path(merge_map_path).exists():
            val_df = apply_class_merge(val_df, label_col, merge_map_path)
        val_df = _filter_known_labels(val_df, label_col, label_map)
        logger.info("Loaded val CSV: %s rows from %s", len(val_df), val_path)
    else:
        stratify_col = train_df[label_col] if label_col in train_df.columns else None
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_split,
            random_state=seed,
            stratify=stratify_col,
        )
        logger.info(
            "Auto-split → train: %d rows, val: %d rows (split=%.2f)",
            len(train_df), len(val_df), val_split,
        )

    train_ds = CXRDataset(
        df=train_df.reset_index(drop=True),
        label_map=label_map,
        image_col=image_col,
        label_col=label_col,
        text_col=text_col,
        transforms=train_transforms,
    )
    val_ds = CXRDataset(
        df=val_df.reset_index(drop=True),
        label_map=label_map,
        image_col=image_col,
        label_col=label_col,
        text_col=text_col,
        transforms=val_transforms,
    )
    return train_ds, val_ds


def build_label_map_from_csv(
    csv_path: Union[str, Path],
    label_col: str = "impression",
    save_path: Optional[Union[str, Path]] = None,
) -> LabelMap:
    """Derive the label map from all unique values in a CSV column."""
    df = _read_csv(csv_path)
    if label_col not in df.columns:
        raise DataLoadError(f"Label column '{label_col}' not found in {csv_path}")
    label_map = LabelMap.from_labels(df[label_col].astype(str).tolist())
    if save_path is not None:
        label_map.save(Path(save_path))
        logger.info("Saved label map (%d classes) → %s", label_map.num_classes, save_path)
    return label_map


def _filter_known_labels(
    df: pd.DataFrame, label_col: str, label_map: LabelMap
) -> pd.DataFrame:
    """Drop rows whose label is not in the label map, with a warning."""
    if label_col not in df.columns:
        return df
    known = set(label_map.str_to_idx.keys())
    mask = df[label_col].astype(str).isin(known)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.warning(
            "Dropping %d rows with labels not in label map (out of %d total).",
            n_dropped, len(df),
        )
    return df[mask].reset_index(drop=True)


def apply_class_merge(
    df: pd.DataFrame,
    label_col: str,
    merge_map_path: Union[str, Path],
) -> pd.DataFrame:
    """Remap labels through a class-merge JSON, collapsing near-identical classes.

    The merge map JSON must have a ``"mapping"`` key: {original_label: merged_label}.
    Rows whose label is not in the mapping are dropped with a warning.

    Args:
        df:             Input dataframe.
        label_col:      Column containing raw label strings.
        merge_map_path: Path to ``class_merge_map.json``.

    Returns:
        New dataframe with ``label_col`` replaced by merged labels.
    """
    import json
    with open(merge_map_path) as fh:
        merge_cfg = json.load(fh)
    mapping: dict[str, str] = merge_cfg["mapping"]

    original = df[label_col].astype(str)
    df = df.copy()
    df[label_col] = original.map(mapping)

    n_unmapped = df[label_col].isna().sum()
    if n_unmapped > 0:
        logger.warning(
            "apply_class_merge: %d rows had labels not in merge map – dropped.", n_unmapped
        )
        df = df.dropna(subset=[label_col])

    df[label_col] = df[label_col].astype(str)
    logger.info(
        "apply_class_merge: %d rows → %d merged classes",
        len(df), df[label_col].nunique(),
    )
    return df.reset_index(drop=True)


def _read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV, raise DataLoadError on failure."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Cannot read CSV '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Multi-label dataset
# ---------------------------------------------------------------------------


class MultiLabelCXRDataset(Dataset):
    """Chest X-ray multi-label dataset.

    Each row carries a binary float vector of shape ``(num_classes,)`` derived
    from pre-computed label columns in the DataFrame (one column per class).

    Args:
        df:          DataFrame with image bytes/paths AND one boolean/int
                     column per pathology class.
        label_cols:  Ordered list of column names that hold binary labels.
        image_col:   Column containing image bytes or file paths.
        transforms:  Optional callable applied to the decoded numpy image array.
        cache_images: Pre-decode all images into RAM.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_cols: list[str],
        image_col: str = "image",
        transforms: Optional[Callable] = None,
        cache_images: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_col = image_col
        self.transforms = transforms
        self._cache: dict[int, np.ndarray] = {}

        self._has_images = image_col in df.columns
        if not self._has_images:
            logger.warning("Column '%s' not found – dataset will return zero images.", image_col)

        if cache_images and self._has_images:
            logger.info("Pre-caching %d images …", len(self.df))
            for i in range(len(self.df)):
                self._cache[i] = self._load_image(i)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        label_vec = torch.tensor(
            [float(row[col]) for col in self.label_cols], dtype=torch.float32
        )

        if self._has_images:
            img_arr = self._cache.get(idx) or self._load_image(idx)
            if self.transforms is not None:
                img_arr = self.transforms(img_arr)
            image_tensor = torch.as_tensor(img_arr, dtype=torch.float32)
        else:
            image_tensor = torch.zeros(1, 1, 1, dtype=torch.float32)

        return {"image": image_tensor, "label": label_vec}

    def class_pos_weights(self) -> torch.Tensor:
        """Return per-class positive weights = (neg_count / pos_count) for BCE loss."""
        weights = []
        n = len(self.df)
        for col in self.label_cols:
            pos = float(self.df[col].sum())
            neg = n - pos
            weights.append(neg / max(pos, 1.0))
        return torch.tensor(weights, dtype=torch.float32)

    def _load_image(self, idx: int) -> np.ndarray:
        return _decode_image_value(self.df.iloc[idx][self.image_col])


def build_multilabel_datasets(
    csv_path: Union[str, Path],
    label_cols: list[str],
    val_csv_path: Optional[Union[str, Path]] = None,
    val_split: float = 0.15,
    image_col: str = "image",
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
    seed: int = 42,
) -> tuple["MultiLabelCXRDataset", "MultiLabelCXRDataset"]:
    """Load CSV(s) and return ``(train_ds, val_ds)`` for multi-label training."""
    from sklearn.model_selection import train_test_split

    train_df = _read_csv(csv_path)
    logger.info("Multilabel train CSV: %d rows from %s", len(train_df), csv_path)

    if val_csv_path is not None:
        val_df = _read_csv(val_csv_path)
        logger.info("Multilabel val CSV: %d rows from %s", len(val_df), val_csv_path)
    else:
        train_df, val_df = train_test_split(
            train_df, test_size=val_split, random_state=seed
        )
        logger.info("Auto-split → train: %d rows, val: %d rows", len(train_df), len(val_df))

    train_ds = MultiLabelCXRDataset(
        df=train_df.reset_index(drop=True),
        label_cols=label_cols,
        image_col=image_col,
        transforms=train_transforms,
    )
    val_ds = MultiLabelCXRDataset(
        df=val_df.reset_index(drop=True),
        label_cols=label_cols,
        image_col=image_col,
        transforms=val_transforms,
    )
    return train_ds, val_ds


def load_and_prepare_dataframe(
    path: Union[str, Path],
    label_col: str,
    label_map: Optional["LabelMap"] = None,
    merge_map_path: Optional[Union[str, Path]] = None,
    tag: str = "CSV",
) -> pd.DataFrame:
    """Central helper: read → optional merge → optional label filter.

    All training, evaluation, calibration, thresholds, and explainability
    entry-points should call this instead of duplicating the three-step pattern.

    Args:
        path:           CSV file to load.
        label_col:      Column containing label strings.
        label_map:      If provided, rows with unknown labels are dropped.
        merge_map_path: If provided and the file exists, apply class merge
                        before label filtering.
        tag:            Short descriptor used in log messages (e.g. "Train",
                        "Val", "Evaluation").

    Returns:
        Cleaned :class:`pd.DataFrame` ready for :class:`CXRDataset`.
    """
    df = _read_csv(path)
    if merge_map_path and Path(merge_map_path).exists():
        logger.info("Applying class merge from %s", merge_map_path)
        df = apply_class_merge(df, label_col, merge_map_path)
    if label_map is not None:
        df = _filter_known_labels(df, label_col, label_map)
    logger.info("%s CSV: %s (%d rows after merge/filter)", tag, path, len(df))
    return df
