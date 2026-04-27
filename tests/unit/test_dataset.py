"""Unit tests for src.train.dataset."""
from __future__ import annotations

import io
import struct

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.common.schemas import LabelMap
from src.train.dataset import CXRDataset, _decode_image_value, build_label_map_from_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


LABELS = ["class_a", "class_b", "class_c"]


def _make_jpeg_bytes(h: int = 32, w: int = 32) -> bytes:
    """Create a minimal valid JPEG bytes object."""
    img = Image.fromarray(
        np.random.randint(0, 255, (h, w), dtype=np.uint8), mode="L"
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def label_map() -> LabelMap:
    return LabelMap.from_labels(LABELS)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    n = 12
    jpegs = [repr(_make_jpeg_bytes()) for _ in range(n)]
    return pd.DataFrame(
        {
            "image": jpegs,
            "impression": [LABELS[i % len(LABELS)] for i in range(n)],
            "findings": [f"findings text {i}" for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# _decode_image_value
# ---------------------------------------------------------------------------


def test_decode_bytes_literal():
    raw_bytes = _make_jpeg_bytes()
    literal_str = repr(raw_bytes)   # b'\xff\xd8...'
    arr = _decode_image_value(literal_str)
    assert arr.ndim == 3                 # CHW
    assert arr.shape[0] == 1            # 1 channel
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_decode_bytes_object():
    raw_bytes = _make_jpeg_bytes()
    arr = _decode_image_value(raw_bytes)
    assert arr.ndim == 3
    assert arr.shape[0] == 1


def test_decode_file_path(tmp_path):
    raw_bytes = _make_jpeg_bytes()
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(raw_bytes)
    arr = _decode_image_value(str(img_file))
    assert arr.shape[0] == 1


def test_decode_invalid_raises():
    from src.common.exceptions import DataLoadError
    with pytest.raises(DataLoadError):
        _decode_image_value(12345)


# ---------------------------------------------------------------------------
# CXRDataset
# ---------------------------------------------------------------------------


def test_dataset_len(sample_df, label_map):
    ds = CXRDataset(df=sample_df, label_map=label_map)
    assert len(ds) == len(sample_df)


def test_dataset_getitem_keys(sample_df, label_map):
    ds = CXRDataset(df=sample_df, label_map=label_map)
    item = ds[0]
    assert "image" in item
    assert "label" in item
    assert "label_str" in item
    assert "findings" in item


def test_dataset_label_range(sample_df, label_map):
    ds = CXRDataset(df=sample_df, label_map=label_map)
    for i in range(len(ds)):
        assert 0 <= ds[i]["label"].item() < label_map.num_classes


def test_dataset_class_weights(sample_df, label_map):
    ds = CXRDataset(df=sample_df, label_map=label_map)
    w = ds.compute_class_weights()
    assert w.shape[0] == label_map.num_classes
    assert (w > 0).all()


def test_dataset_weighted_sampler(sample_df, label_map):
    ds = CXRDataset(df=sample_df, label_map=label_map)
    sampler = ds.make_weighted_sampler()
    assert len(sampler) == len(ds)


# ---------------------------------------------------------------------------
# LabelMap
# ---------------------------------------------------------------------------


def test_label_map_roundtrip(tmp_path):
    lm = LabelMap.from_labels(LABELS)
    path = tmp_path / "label_map.json"
    lm.save(path)
    loaded = LabelMap.load(path)
    assert loaded.num_classes == lm.num_classes
    for lbl in LABELS:
        assert loaded.encode(lbl) == lm.encode(lbl)


def test_label_map_unknown_raises():
    lm = LabelMap.from_labels(LABELS)
    with pytest.raises(KeyError):
        lm.encode("nonexistent_class")
