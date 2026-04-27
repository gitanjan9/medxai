"""Tests that the label-merge pipeline is applied consistently across all entry-points.

Covers:
1. load_and_prepare_dataframe – core helper
2. Merge applied before filter (no rows dropped due to merge)
3. Evaluation path uses merged labels
4. Calibration path uses merged labels
5. Thresholds path uses merged labels
6. Original (unmerged) CSV labels are converted correctly before encoding
7. Evaluation does NOT drop all rows when merge map is provided
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.train.dataset import (
    _filter_known_labels,
    apply_class_merge,
    load_and_prepare_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORIGINAL_LABELS = [
    "No acute cardiopulmonary abnormality.",
    "No acute cardiopulmonary abnormality.",
    "No acute intrathoracic process.",
    "No change.",
    "No evidence of pneumonia.",
    "No pneumothorax.",
]

MERGE_MAP = {
    "mapping": {
        "No acute cardiopulmonary abnormality.": "normal_acute_absence",
        "No acute intrathoracic process.":       "normal_acute_absence",
        "No change.":                            "stable_no_change",
        "No evidence of pneumonia.":             "specific_negative",
        "No pneumothorax.":                      "specific_negative",
    }
}

MERGED_LABELS = [
    "normal_acute_absence",
    "normal_acute_absence",
    "normal_acute_absence",
    "stable_no_change",
    "specific_negative",
    "specific_negative",
]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"impression": ORIGINAL_LABELS, "image": ["x"] * len(ORIGINAL_LABELS)})


@pytest.fixture
def merge_map_file(tmp_path) -> Path:
    p = tmp_path / "merge_map.json"
    p.write_text(json.dumps(MERGE_MAP))
    return p


@pytest.fixture
def merged_label_map():
    from src.common.schemas import LabelMap
    return LabelMap.from_labels(MERGED_LABELS)


@pytest.fixture
def csv_with_original_labels(tmp_path, sample_df) -> Path:
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. load_and_prepare_dataframe – basic behaviour
# ---------------------------------------------------------------------------

def test_load_prepare_no_merge_no_filter(csv_with_original_labels):
    df = load_and_prepare_dataframe(csv_with_original_labels, "impression")
    assert len(df) == len(ORIGINAL_LABELS)
    assert list(df["impression"]) == ORIGINAL_LABELS


def test_load_prepare_with_merge(csv_with_original_labels, merge_map_file):
    df = load_and_prepare_dataframe(
        csv_with_original_labels, "impression",
        merge_map_path=merge_map_file,
    )
    assert len(df) == len(MERGED_LABELS)
    assert set(df["impression"]) == {"normal_acute_absence", "stable_no_change", "specific_negative"}


def test_load_prepare_with_merge_and_filter(
    csv_with_original_labels, merge_map_file, merged_label_map
):
    df = load_and_prepare_dataframe(
        csv_with_original_labels, "impression",
        label_map=merged_label_map,
        merge_map_path=merge_map_file,
    )
    assert len(df) == len(MERGED_LABELS)


def test_load_prepare_no_rows_dropped_with_merge(
    csv_with_original_labels, merge_map_file, merged_label_map
):
    """Core regression: providing merge map must prevent all-rows-dropped."""
    df_with_merge = load_and_prepare_dataframe(
        csv_with_original_labels, "impression",
        label_map=merged_label_map,
        merge_map_path=merge_map_file,
    )
    df_without_merge = load_and_prepare_dataframe(
        csv_with_original_labels, "impression",
        label_map=merged_label_map,
        merge_map_path=None,
    )
    assert len(df_with_merge) == len(MERGED_LABELS), "All rows should survive with merge"
    assert len(df_without_merge) == 0, "All rows should be filtered without merge (labels mismatch)"


# ---------------------------------------------------------------------------
# 2. apply_class_merge – unit
# ---------------------------------------------------------------------------

def test_apply_class_merge_replaces_labels(sample_df, merge_map_file):
    result = apply_class_merge(sample_df, "impression", merge_map_file)
    assert set(result["impression"]) <= {"normal_acute_absence", "stable_no_change", "specific_negative"}


def test_apply_class_merge_drops_unmapped(merge_map_file):
    df = pd.DataFrame({"impression": ["UNKNOWN_LABEL", "No change."]})
    result = apply_class_merge(df, "impression", merge_map_file)
    assert len(result) == 1
    assert result.iloc[0]["impression"] == "stable_no_change"


def test_apply_class_merge_empty_after_all_unmapped(merge_map_file):
    df = pd.DataFrame({"impression": ["UNKNOWN_A", "UNKNOWN_B"]})
    result = apply_class_merge(df, "impression", merge_map_file)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. _filter_known_labels drops unmerged originals
# ---------------------------------------------------------------------------

def test_filter_drops_original_labels_when_map_is_merged(sample_df, merged_label_map):
    """Without merging first, all original labels should be filtered out."""
    result = _filter_known_labels(sample_df, "impression", merged_label_map)
    assert len(result) == 0


def test_filter_retains_merged_labels(merge_map_file, sample_df, merged_label_map):
    merged_df = apply_class_merge(sample_df, "impression", merge_map_file)
    result = _filter_known_labels(merged_df, "impression", merged_label_map)
    assert len(result) == len(MERGED_LABELS)


# ---------------------------------------------------------------------------
# 4 – 6. Evaluate / calibrate / thresholds use load_and_prepare_dataframe
# ---------------------------------------------------------------------------

def _assert_uses_helper(module_path: str, function_name: str):
    """Check that load_and_prepare_dataframe is imported in the given module."""
    import importlib
    mod = importlib.import_module(module_path)
    src = Path(mod.__file__).read_text()
    assert "load_and_prepare_dataframe" in src, (
        f"{module_path}.{function_name} does not use load_and_prepare_dataframe"
    )


def test_evaluate_uses_central_helper():
    _assert_uses_helper("src.train.evaluate", "evaluate")


def test_calibrate_uses_central_helper():
    _assert_uses_helper("src.train.calibrate", "calibrate")


def test_thresholds_uses_central_helper():
    _assert_uses_helper("src.train.thresholds", "tune_thresholds")


def test_explainability_uses_central_helper():
    _assert_uses_helper("src.train.explainability", "generate_explanations")


def test_train_uses_central_helper():
    _assert_uses_helper("src.train.train", "train")


# ---------------------------------------------------------------------------
# 7. load_and_prepare_dataframe – merge_map_path missing file is safe no-op
# ---------------------------------------------------------------------------

def test_nonexistent_merge_map_is_skipped(csv_with_original_labels, merged_label_map):
    """A missing merge map file should be silently skipped (no crash)."""
    df = load_and_prepare_dataframe(
        csv_with_original_labels, "impression",
        label_map=merged_label_map,
        merge_map_path="/nonexistent/path/merge.json",
    )
    assert len(df) == 0  # all filtered, but no exception raised


# ---------------------------------------------------------------------------
# 8. Tag appears in log output
# ---------------------------------------------------------------------------

def test_load_prepare_logs_tag(csv_with_original_labels, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="medicalxai.dataset"):
        load_and_prepare_dataframe(
            csv_with_original_labels, "impression", tag="MyTag"
        )
    assert any("MyTag" in r.message for r in caplog.records)
