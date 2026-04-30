"""Convert a torchxrayvision PathologyResult into a PrimaryPrediction.

Decision pipeline (in order of priority):
  1. Positive   – score >= class-specific threshold in TXRV_THRESHOLDS.
  2. Uncertain  – no positives AND (top score < UNCERTAIN_TOP_SCORE OR
                  top-5 spread < UNCERTAIN_SPREAD).  Never overrides a
                  confirmed positive finding.
  3. Review     – no positives, not uncertain, but score >= REVIEW_MIN_SCORE.
  4. Negative   – nothing at or above REVIEW_MIN_SCORE.

Suppression rule: scores < REVIEW_MIN_SCORE appear only in all_scores,
never in positive_findings or review_findings.
"""
from __future__ import annotations

from src.config.txrv_thresholds import (
    BROAD_ABNORMALITY_MIN_CLASSES,
    DEFAULT_THRESHOLD,
    REVIEW_MIN_SCORE,
    UNCERTAIN_SPREAD,
    UNCERTAIN_TOP_SCORE,
    get_threshold,
)
from src.serve.schemas.response import PathologyFinding, PrimaryPrediction
from src.serve.services.clinical_ranker import build_clinical_output
from src.serve.services.explanation_engine import build_explanation
from src.serve.services.pathology_detector import PathologyResult
from src.serve.services.pleural_analyzer import PleuralFeatures

_THRESHOLD_VERSION = "txrv_densenet121_class_thresholds_v1"
_HIGH_CONFIDENCE_MIN = 0.85   # score >= this within a positive → "high" band

# No Finding gate: when the model's own No Finding score is >= this value,
# only report findings that exceed _HIGH_CONFIDENCE_MIN (very strong signal).
# This uses the model's own normality prediction to suppress false positives
# on near-normal CXRs without touching any per-class thresholds.
_NO_FINDING_GATE = 0.50

_RESPIRATORY_CLUSTER = frozenset([
    "Pneumonia", "Infiltration", "Lung Opacity",
    "Consolidation", "Edema", "Atelectasis",
])
_CLUSTER_OVERRIDE_MIN = 3   # 3+ respiratory classes firing → spread check bypassed


_PTX_RESTORE_THRESHOLD   = 0.35   # ptx_evidence >= this → restore PTX score
_PTX_BASE_MIN            = 0.42   # base model score must reach this before pleural boost applies
_PTX_NF_SUPPRESS         = 0.35   # no_finding_score >= this → suppress PTX (mutual exclusion)
_EMPH_BOOST_THRESHOLD    = 0.55   # emphysema_evidence >= this → boost Emphysema
_BULLOUS_BOOST_THRESHOLD = 0.45   # bullous_evidence >= this → flag Bullous
_BULLOUS_EMPH_BOOST      = 0.05   # added to Emphysema score when bullous pattern detected
_EMPH_BOOST              = 0.08   # added to Emphysema score when hyperinflation confirmed
_PTX_RESTORE             = 0.12   # added back to PTX score when pleural line confirmed


def _keep_finding(
    name: str,
    score: float,
    ptx_suppressed: bool,
    no_finding_gate_active: bool,
) -> bool:
    """Return False when a finding should be suppressed by normality gates."""
    if name == "Pneumothorax" and ptx_suppressed:
        return False
    if no_finding_gate_active and score < _HIGH_CONFIDENCE_MIN:
        return False
    return True


def _apply_pleural_corrections(
    scores: dict[str, float],
    feats: PleuralFeatures,
) -> tuple[dict[str, float], list[str]]:
    """Adjust raw scores using image-derived pleural features.

    Runs BEFORE conflict rules so the corrected scores feed into them.
    """
    adj = dict(scores)
    log: list[str] = []

    # 1. Pneumothorax: if pleural line + peripheral dark zone → restore score.
    # Guard: only boost if base model score >= 0.42 (meaningful PTX signal).
    # This prevents normal lung margins from triggering the restore on near-normal CXRs.
    if feats.ptx_evidence >= _PTX_RESTORE_THRESHOLD:
        old = adj.get("Pneumothorax", 0.0)
        if old >= _PTX_BASE_MIN:
            adj["Pneumothorax"] = min(1.0, old + _PTX_RESTORE * feats.ptx_evidence)
            log.append(
                f"Pneumothorax restored {old:.3f}→{adj['Pneumothorax']:.3f} "
                f"(ptx_evidence={feats.ptx_evidence:.3f})"
            )
        else:
            log.append(
                f"Pneumothorax boost skipped: base {old:.3f} < {_PTX_BASE_MIN} "
                f"(ptx_evidence={feats.ptx_evidence:.3f})"
            )

    # 2. Emphysema: flat diaphragm → boost score
    if feats.emphysema_evidence >= _EMPH_BOOST_THRESHOLD:
        old = adj.get("Emphysema", 0.0)
        adj["Emphysema"] = min(1.0, old + _EMPH_BOOST * feats.emphysema_evidence)
        log.append(
            f"Emphysema boosted {old:.3f}→{adj['Emphysema']:.3f} "
            f"(emphysema_evidence={feats.emphysema_evidence:.3f})"
        )

    # 3. Bullous: focal avascular zones → mark with Emphysema + note
    if feats.bullous_evidence >= _BULLOUS_BOOST_THRESHOLD and (
        feats.ptx_evidence < _PTX_RESTORE_THRESHOLD
    ):
        old = adj.get("Emphysema", 0.0)
        adj["Emphysema"] = min(1.0, old + _BULLOUS_EMPH_BOOST * feats.bullous_evidence)
        log.append(
            f"Bullous pattern detected (emphysema_evidence={feats.emphysema_evidence:.3f} "
            f"focal_avascular={feats.focal_avascular_score:.3f})"
        )

    return adj, log


def txrv_to_primary_prediction(
    result: PathologyResult,
    pleural_feats: PleuralFeatures | None = None,
) -> PrimaryPrediction:
    """Build a ``PrimaryPrediction`` from a torchxrayvision ``PathologyResult``.

    Args:
        result: Output of ``XrvPathologyDetector.predict()``.

    Returns:
        ``PrimaryPrediction`` reflecting class-specific thresholds and
        uncertainty suppression.
    """
    pleural_log: list[str] = []
    base_scores = result.scores
    if pleural_feats is not None:
        base_scores, pleural_log = _apply_pleural_corrections(result.scores, pleural_feats)

    clinical = build_clinical_output(base_scores)
    adj = clinical.adjusted_scores

    no_finding_score = result.scores.get("No Finding", 0.0)
    no_finding_gate_active = no_finding_score >= _NO_FINDING_GATE

    # PTX anti-correlation: Pneumothorax and No Finding are clinically mutually exclusive.
    # Suppress PTX from findings when the model assigns any meaningful No Finding score.
    ptx_suppressed = no_finding_score >= _PTX_NF_SUPPRESS

    positive_findings: list[PathologyFinding] = [
        PathologyFinding(name=f.name, score=round(f.adjusted_score, 4))
        for f in clinical.all_positives()
        if _keep_finding(f.name, f.adjusted_score, ptx_suppressed, no_finding_gate_active)
    ]
    review_findings: list[PathologyFinding] = [
        PathologyFinding(name=f.name, score=round(f.adjusted_score, 4))
        for f in clinical.all_review()
        if _keep_finding(f.name, f.adjusted_score, ptx_suppressed, no_finding_gate_active)
        and not no_finding_gate_active
    ]

    non_nf = sorted(
        [(k, v) for k, v in adj.items() if k != "No Finding"],
        key=lambda x: x[1], reverse=True,
    )
    top_score = non_nf[0][1] if non_nf else 0.0
    top_5_scores = [s for _, s in non_nf[:5]]
    spread = (max(top_5_scores) - min(top_5_scores)) if len(top_5_scores) >= 2 else 1.0

    broad_hits = sum(
        1 for name, score in adj.items()
        if name != "No Finding" and score >= REVIEW_MIN_SCORE
    )
    respiratory_hits = sum(
        1 for name, score in adj.items()
        if name in _RESPIRATORY_CLUSTER and score >= REVIEW_MIN_SCORE
    )

    if positive_findings:
        label = positive_findings[0].name
        score = positive_findings[0].score
        threshold = get_threshold(label)
        status = "positive"
        band = "high" if score >= _HIGH_CONFIDENCE_MIN else "medium"
        if respiratory_hits >= _CLUSTER_OVERRIDE_MIN:
            review_reason = "respiratory_cluster_confirmed"
        elif broad_hits >= BROAD_ABNORMALITY_MIN_CLASSES:
            review_reason = f"broad_pattern_{broad_hits}_classes"
        else:
            review_reason = ""

    elif review_findings:
        # Surface review findings BEFORE uncertainty gate — real findings must not be suppressed.
        label = review_findings[0].name
        score = review_findings[0].score
        threshold = get_threshold(label)
        status = "review_required"
        band = "medium"
        if broad_hits >= BROAD_ABNORMALITY_MIN_CLASSES:
            review_reason = (
                f"Diffuse abnormality pattern: {broad_hits} pathology classes active. "
                "Scores below individual thresholds but broad activation warrants review."
            )
        else:
            review_reason = "below_class_threshold"

    else:
        # Nothing at or above REVIEW_MIN_SCORE — now check for uncertainty vs negative.
        is_uncertain = top_score < UNCERTAIN_TOP_SCORE or spread < UNCERTAIN_SPREAD
        if is_uncertain:
            label = "No Confident Finding"
            score = round(top_score, 4)
            threshold = UNCERTAIN_TOP_SCORE
            status = "likely_normal_or_uncertain"
            band = "low"
            review_reason = (
                "No confident abnormality detected. "
                "Clinical review recommended if symptoms exist."
            )
        else:
            label = "No Finding"
            score = round(result.no_finding_score, 4)
            threshold = DEFAULT_THRESHOLD
            status = "negative"
            band = "low"
            review_reason = ""

    explanation = build_explanation(
        primary_label=label,
        primary_score=round(score, 4),
        decision=status,
        raw_scores=result.scores,
        final_scores=adj,
        pleural_log=pleural_log,
        conflict_log=clinical.conflict_log,
        pleural_feats_dict=pleural_feats.to_dict() if pleural_feats is not None else None,
    )

    return PrimaryPrediction(
        label=label,
        raw_score=round(score, 4),
        calibrated_score=round(score, 4),
        all_scores=adj,
        decision=status,
        confidence_band=band,
        review_reason=review_reason,
        threshold_version=_THRESHOLD_VERSION,
        threshold=round(threshold, 4),
        positive_findings=positive_findings,
        review_findings=review_findings,
        clinical_groups=clinical.to_api_groups(),
        conflict_log=pleural_log + clinical.conflict_log,
        model_explanation=explanation.to_dict(),
    )
