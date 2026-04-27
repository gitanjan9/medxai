"""Unit tests for src/train/hierarchical.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.train.hierarchical import (
    HierarchicalPipeline,
    HierarchicalResult,
    _STAGE2_TRIGGER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STAGE1_CLASSES = ["normal_acute_absence", "stable_no_change", "specific_negative"]
_STAGE2_CLASSES = ["no_acute_cardiopulmonary", "no_acute_intrathoracic"]


def _make_model(num_classes: int, fixed_logit_idx: int) -> nn.Module:
    """Tiny model that always returns max logit at fixed_logit_idx."""
    class FixedModel(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], num_classes)
            logits[:, fixed_logit_idx] = 10.0
            return logits
    return FixedModel()


@pytest.fixture
def pipeline_triggers_stage2():
    """Pipeline where stage1 always predicts normal_acute_absence → triggers stage2."""
    s1 = _make_model(3, 0)  # idx 0 = normal_acute_absence
    s2 = _make_model(2, 1)  # idx 1 = no_acute_intrathoracic
    return HierarchicalPipeline(
        stage1_model=s1, stage1_classes=_STAGE1_CLASSES,
        stage2_model=s2, stage2_classes=_STAGE2_CLASSES,
        device=torch.device("cpu"),
    )


@pytest.fixture
def pipeline_stable():
    """Pipeline where stage1 always predicts stable_no_change → no stage2."""
    s1 = _make_model(3, 1)  # idx 1 = stable_no_change
    s2 = _make_model(2, 0)  # irrelevant
    return HierarchicalPipeline(
        stage1_model=s1, stage1_classes=_STAGE1_CLASSES,
        stage2_model=s2, stage2_classes=_STAGE2_CLASSES,
        device=torch.device("cpu"),
    )


@pytest.fixture
def dummy_image():
    return torch.randn(1, 1, 224, 224)


# ---------------------------------------------------------------------------
# Tests: predict_single path through stage2
# ---------------------------------------------------------------------------

def test_stage2_triggered(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    assert result.stage1_class == "normal_acute_absence"
    assert result.stage2_class == "no_acute_intrathoracic"
    assert result.final_class == "no_acute_intrathoracic"
    assert result.stage2_probs is not None
    assert set(result.stage2_probs.keys()) == set(_STAGE2_CLASSES)


def test_stage2_not_triggered(pipeline_stable, dummy_image):
    result = pipeline_stable.predict_single(dummy_image)
    assert result.stage1_class == "stable_no_change"
    assert result.stage2_class is None
    assert result.stage2_probs is None
    assert result.final_class == "stable_no_change"


# ---------------------------------------------------------------------------
# Tests: stage1 probs present and sum to ~1
# ---------------------------------------------------------------------------

def test_stage1_probs_complete(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    assert set(result.stage1_probs.keys()) == set(_STAGE1_CLASSES)
    total = sum(result.stage1_probs.values())
    assert abs(total - 1.0) < 0.01


def test_stage2_probs_complete(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    assert set(result.stage2_probs.keys()) == set(_STAGE2_CLASSES)
    total = sum(result.stage2_probs.values())
    assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: top3 structure
# ---------------------------------------------------------------------------

def test_top3_length(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    assert 1 <= len(result.top3) <= 3


def test_top3_sorted_descending(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    probs = [p for _, p in result.top3]
    assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Tests: as_dict serialisable
# ---------------------------------------------------------------------------

def test_as_dict_serialisable(pipeline_triggers_stage2, dummy_image):
    result = pipeline_triggers_stage2.predict_single(dummy_image)
    d = result.as_dict()
    serialised = json.dumps(d)
    parsed = json.loads(serialised)
    assert parsed["final_class"] == result.final_class
    assert parsed["stage2"]["class"] == result.stage2_class


def test_as_dict_no_stage2(pipeline_stable, dummy_image):
    result = pipeline_stable.predict_single(dummy_image)
    d = result.as_dict()
    assert d["stage2"] is None


# ---------------------------------------------------------------------------
# Tests: predict_batch
# ---------------------------------------------------------------------------

def test_predict_batch_length(pipeline_stable):
    batch = torch.randn(4, 1, 224, 224)
    results = pipeline_stable.predict_batch(batch)
    assert len(results) == 4
    for r in results:
        assert isinstance(r, HierarchicalResult)


def test_predict_batch_single_equiv(pipeline_triggers_stage2):
    img = torch.randn(1, 1, 224, 224)
    single = pipeline_triggers_stage2.predict_single(img)
    batch = pipeline_triggers_stage2.predict_batch(img)
    assert len(batch) == 1
    assert batch[0].final_class == single.final_class


# ---------------------------------------------------------------------------
# Tests: 3-dim input (no batch dim) is auto-unsqueezed
# ---------------------------------------------------------------------------

def test_single_3dim_input(pipeline_stable):
    img = torch.randn(1, 224, 224)  # [C, H, W] – no batch dim
    result = pipeline_stable.predict_single(img)
    assert result.final_class == "stable_no_change"


# ---------------------------------------------------------------------------
# Tests: temperature scaling changes probs but not argmax for extreme logits
# ---------------------------------------------------------------------------

def test_temperature_scaling_applied(dummy_image):
    s1 = _make_model(3, 0)
    s2 = _make_model(2, 1)
    p_t1 = HierarchicalPipeline(
        stage1_model=s1, stage1_classes=_STAGE1_CLASSES,
        stage2_model=s2, stage2_classes=_STAGE2_CLASSES,
        temperature1=1.0, device=torch.device("cpu"),
    )
    p_t2 = HierarchicalPipeline(
        stage1_model=s1, stage1_classes=_STAGE1_CLASSES,
        stage2_model=s2, stage2_classes=_STAGE2_CLASSES,
        temperature1=2.0, device=torch.device("cpu"),
    )
    r1 = p_t1.predict_single(dummy_image)
    r2 = p_t2.predict_single(dummy_image)
    # argmax stays the same (extreme logits)
    assert r1.stage1_class == r2.stage1_class
    # but probs differ
    assert r1.stage1_probs != r2.stage1_probs
