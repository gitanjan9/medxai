"""Model explanation engine for MedicalXAI.

Builds a full transparent explanation of WHY the model reached its conclusion,
including:
  1. Evidence chain     – score at each processing stage for the winning class
  2. Ruled-out labels   – competing hypotheses and what suppressed them
  3. Model biases       – known limitations of torchxrayvision DenseNet-121
  4. Clinical narrative – plain-English paragraph summary

This runs on the outputs already computed by the clinical ranker and pleural
analyzer — no extra inference cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config.class_conflicts import CONFLICT_RULES
from src.config.txrv_thresholds import REVIEW_MIN_SCORE, get_threshold


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvidenceStep:
    stage: str          # "DenseNet raw" | "Pleural correction" | "Conflict penalty"
    value_before: float
    value_after: float
    delta: float        # value_after - value_before
    reason: str         # human-readable explanation


@dataclass
class RuledOutHypothesis:
    label: str
    raw_score: float
    final_score: float
    suppressed_by: str   # which rule / why
    delta: float         # how much was removed


@dataclass
class ModelBias:
    name: str
    relevance: str       # "high" | "medium" | "low"
    description: str
    impact_on_this_case: str


@dataclass
class ModelExplanation:
    primary_label: str
    primary_score: float
    decision: str

    evidence_chain: list[EvidenceStep]
    ruled_out: list[RuledOutHypothesis]
    model_biases: list[ModelBias]
    clinical_narrative: str

    def to_dict(self) -> dict:
        return {
            "primary_label": self.primary_label,
            "primary_score": round(self.primary_score, 4),
            "decision": self.decision,
            "evidence_chain": [
                {
                    "stage": e.stage,
                    "value_before": round(e.value_before, 4),
                    "value_after": round(e.value_after, 4),
                    "delta": round(e.delta, 4),
                    "reason": e.reason,
                }
                for e in self.evidence_chain
            ],
            "ruled_out": [
                {
                    "label": r.label,
                    "raw_score": round(r.raw_score, 4),
                    "final_score": round(r.final_score, 4),
                    "suppressed_by": r.suppressed_by,
                    "delta": round(r.delta, 4),
                }
                for r in self.ruled_out
            ],
            "model_biases": [
                {
                    "name": b.name,
                    "relevance": b.relevance,
                    "description": b.description,
                    "impact_on_this_case": b.impact_on_this_case,
                }
                for b in self.model_biases
            ],
            "clinical_narrative": self.clinical_narrative,
        }


# ── Known model biases ────────────────────────────────────────────────────────

_ALL_BIASES: list[dict] = [
    {
        "name": "NIH CXR14 label noise",
        "classes": None,  # applies broadly
        "description": (
            "torchxrayvision DenseNet-121 was trained on NIH CXR14 dataset whose labels "
            "were extracted by NLP from radiology reports — not direct radiologist annotation. "
            "Estimated label error rate is 15–20%."
        ),
        "high_for": None,
    },
    {
        "name": "No Finding class imbalance",
        "classes": None,
        "description": (
            "~53% of NIH CXR14 images are labeled 'No Finding'. The model is biased toward "
            "predicting normal, which causes systematic underscoring of subtle pathologies."
        ),
        "high_for": None,
    },
    {
        "name": "Fibrosis / Emphysema texture confusion",
        "classes": ["Fibrosis", "Emphysema"],
        "description": (
            "Both fibrosis and emphysema alter lung parenchymal texture in ways that share "
            "mid-level CNN features. The model frequently co-activates both classes on images "
            "with diffuse lung abnormality. A post-hoc conflict penalty corrects this."
        ),
        "high_for": ["Fibrosis", "Emphysema"],
    },
    {
        "name": "Bullous disease not in training vocabulary",
        "classes": ["Emphysema", "Pneumothorax"],
        "description": (
            "Bullous emphysema is not a separate class in CXR14. The model represents it "
            "as a mixture of Emphysema and Pneumothorax signals. Pleural-line detection is "
            "used post-hoc to distinguish true pneumothorax from bullae."
        ),
        "high_for": ["Emphysema", "Pneumothorax"],
    },
    {
        "name": "Score clustering at 60–65%",
        "classes": None,
        "description": (
            "When the model is uncertain, it outputs near-identical softmax probabilities "
            "(60–65%) for many classes simultaneously — a known DenseNet artefact on "
            "out-of-distribution or ambiguous inputs. This is not meaningful multi-label "
            "detection; it reflects epistemic uncertainty."
        ),
        "high_for": None,
    },
    {
        "name": "AP vs PA view sensitivity",
        "classes": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
        "description": (
            "Cardiac silhouette appears larger on AP (supine/portable) views due to beam "
            "divergence. The model does not receive view metadata, causing systematic "
            "over-prediction of Cardiomegaly on AP films."
        ),
        "high_for": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    },
    {
        "name": "Resolution downsampling to 224px",
        "classes": ["Nodule", "Fracture"],
        "description": (
            "torchxrayvision resizes input to 224×224 px, causing loss of fine detail. "
            "Small nodules (<10mm) and subtle rib fractures may be missed or mis-scored "
            "due to this resolution limitation."
        ),
        "high_for": ["Nodule", "Fracture"],
    },
    {
        "name": "Demographic / scanner bias",
        "classes": None,
        "description": (
            "Training data originates from US hospital populations. Performance may degrade "
            "on images from different geographic populations, scanner types (CR vs DR), "
            "or positioning standards."
        ),
        "high_for": None,
    },
]


def _select_biases(primary_label: str, raw_scores: dict[str, float]) -> list[ModelBias]:
    """Pick the most relevant biases for this prediction."""
    result: list[ModelBias] = []

    # Detect score-clustering
    non_nf = [s for k, s in raw_scores.items() if k != "No Finding"]
    high_count = sum(1 for s in non_nf if 0.58 <= s <= 0.67)
    clustering = high_count >= 5

    for b in _ALL_BIASES:
        relevant = b["classes"] is None or primary_label in (b["classes"] or [])
        relevance = "low"
        impact = "Minimal impact expected for this finding."

        if b["name"] == "NIH CXR14 label noise":
            relevance = "medium"
            impact = (
                f"The raw score for {primary_label} ({raw_scores.get(primary_label, 0):.0%}) "
                "may be slightly inflated or deflated due to NLP label noise in training."
            )

        elif b["name"] == "No Finding class imbalance":
            relevance = "high" if raw_scores.get(primary_label, 1.0) < 0.70 else "medium"
            impact = (
                "Because the model is biased toward normal, the score for "
                f"{primary_label} is likely an underestimate of true pathology probability."
            )

        elif b["name"] == "Score clustering at 60–65%" and clustering:
            relevance = "high"
            impact = (
                f"{high_count} classes are scoring 58–67% simultaneously. "
                "This is the DenseNet uncertainty artefact — the model is not confidently "
                "detecting multiple independent pathologies; it is expressing doubt."
            )

        elif b["name"] == "Fibrosis / Emphysema texture confusion" and primary_label in (
            "Fibrosis", "Emphysema"
        ):
            relevance = "high"
            impact = (
                f"Fibrosis score was conflict-penalised when Emphysema was the driver "
                "(or vice versa). The winning label ({primary_label}) survived because its "
                "score remained highest after post-hoc correction."
            )

        elif b["name"] == "Bullous disease not in training vocabulary" and primary_label in (
            "Emphysema", "Pneumothorax"
        ):
            relevance = "high"
            impact = (
                "If the X-ray shows bullous emphysema, the model may split its vote between "
                "Emphysema and Pneumothorax. Pleural-line analysis is used to break the tie."
            )

        elif b["classes"] and primary_label in b["classes"]:
            relevance = "medium"
            impact = b["description"]

        else:
            if not relevant:
                continue
            relevance = "low"

        result.append(ModelBias(
            name=b["name"],
            relevance=relevance,
            description=b["description"],
            impact_on_this_case=impact,
        ))

    result.sort(key=lambda b: {"high": 0, "medium": 1, "low": 2}[b.relevance])
    return result


# ── Evidence chain builder ────────────────────────────────────────────────────

def _build_evidence_chain(
    primary_label: str,
    raw_scores: dict[str, float],
    pleural_log: list[str],
    conflict_log: list[str],
    final_score: float,
) -> list[EvidenceStep]:
    """Trace score transformation for the primary label from raw → final."""
    chain: list[EvidenceStep] = []
    raw = raw_scores.get(primary_label, final_score)

    chain.append(EvidenceStep(
        stage="DenseNet-121 raw output",
        value_before=raw,
        value_after=raw,
        delta=0.0,
        reason=(
            f"torchxrayvision forward pass on 224×224 grayscale input. "
            f"Sigmoid activation produces class probability in [0,1]. "
            f"Raw score for {primary_label} = {raw:.1%}."
        ),
    ))

    # Pleural corrections
    after_pleural = raw
    for entry in pleural_log:
        if primary_label in entry:
            parts = entry.split("→")
            if len(parts) == 2:
                try:
                    before_str = parts[0].split()[-1]
                    after_str = parts[1].split()[0]
                    before_val = float(before_str)
                    after_val = float(after_str.rstrip(")"))
                    after_pleural = after_val
                    chain.append(EvidenceStep(
                        stage="Pleural-line / vascular analysis correction",
                        value_before=before_val,
                        value_after=after_val,
                        delta=after_val - before_val,
                        reason=entry,
                    ))
                except ValueError:
                    pass

    # Conflict penalties
    after_conflict = after_pleural
    for entry in conflict_log:
        if primary_label in entry and "→" in entry:
            parts = entry.split("→")
            if len(parts) == 2:
                try:
                    before_str = parts[0].split()[-1]
                    after_str = parts[1].split()[0]
                    before_val = float(before_str)
                    after_val = float(after_str.rstrip(")"))
                    after_conflict = after_val
                    chain.append(EvidenceStep(
                        stage="Clinical conflict penalty",
                        value_before=before_val,
                        value_after=after_val,
                        delta=after_val - before_val,
                        reason=entry,
                    ))
                except ValueError:
                    pass

    # Threshold comparison
    thr = get_threshold(primary_label)
    chain.append(EvidenceStep(
        stage="Threshold decision",
        value_before=final_score,
        value_after=final_score,
        delta=0.0,
        reason=(
            f"Class-specific threshold for {primary_label} = {thr:.1%}. "
            f"Final score {final_score:.1%} is "
            f"{'above' if final_score >= thr else 'below'} threshold → "
            f"{'POSITIVE' if final_score >= thr else 'REVIEW REQUIRED'}."
        ),
    ))

    return chain


# ── Ruled-out builder ─────────────────────────────────────────────────────────

def _build_ruled_out(
    primary_label: str,
    raw_scores: dict[str, float],
    final_scores: dict[str, float],
    conflict_log: list[str],
    pleural_log: list[str],
) -> list[RuledOutHypothesis]:
    """List classes that started high but were suppressed."""
    result: list[RuledOutHypothesis] = []
    for cls, raw in sorted(raw_scores.items(), key=lambda x: x[1], reverse=True):
        if cls == primary_label or cls == "No Finding":
            continue
        final = final_scores.get(cls, raw)
        if raw < REVIEW_MIN_SCORE:
            continue
        if final >= REVIEW_MIN_SCORE:
            continue   # still active — not ruled out
        delta = final - raw

        # Find suppression reason
        reason = "Score fell below review threshold after adjustments."
        for entry in (pleural_log + conflict_log):
            if cls in entry:
                reason = entry
                break
        else:
            for rule in CONFLICT_RULES:
                if rule.target == cls and raw_scores.get(rule.driver, 0) >= rule.driver_trigger:
                    driver_score = raw_scores.get(rule.driver, 0.0)
                    reason = (
                        f"Conflict rule: {rule.driver} ({driver_score:.1%}) suppressed "
                        f"{cls} by {rule.penalty:.0%}×{driver_score:.1%} = "
                        f"{rule.penalty * driver_score:.3f}."
                    )
                    break

        result.append(RuledOutHypothesis(
            label=cls,
            raw_score=raw,
            final_score=final,
            suppressed_by=reason,
            delta=delta,
        ))

    return sorted(result, key=lambda r: r.raw_score, reverse=True)[:6]


# ── Narrative generator ───────────────────────────────────────────────────────

_DECISION_PHRASES = {
    "positive": "confirmed positive finding",
    "review_required": "finding requiring clinical review",
    "likely_normal_or_uncertain": "uncertain / likely normal result",
    "negative": "negative (no significant finding)",
}

_CLASS_CLINICAL_DESC = {
    "Emphysema": (
        "Emphysema causes permanent destruction of alveolar walls, resulting in "
        "hyperinflated lung fields, flattened diaphragm, and reduced vascular markings. "
        "The model detects this through low-density texture across both lung fields "
        "combined with diaphragm position analysis."
    ),
    "Pneumothorax": (
        "Pneumothorax is free air in the pleural space, seen as a peripheral avascular "
        "zone bounded by a visible visceral pleural line. The model detects this through "
        "edge density at the lung apex and absence of vessel markings in a wedge-shaped region."
    ),
    "Pneumonia": (
        "Pneumonia appears as focal consolidation — increased airspace density where alveoli "
        "fill with fluid or exudate. The model detects the resulting opacity as "
        "high-attenuation regions with air bronchograms or lobar distribution."
    ),
    "Fibrosis": (
        "Pulmonary fibrosis shows as reticular (net-like) increased density, often basal "
        "and peripheral. The model associates coarse texture patterns and volume loss with "
        "this class."
    ),
    "Atelectasis": (
        "Atelectasis is lung collapse — partial or complete. It appears as increased density "
        "with volume loss, often with mediastinal shift. The model keys on linear or lobar "
        "opacity patterns with crowding of structures."
    ),
    "Consolidation": (
        "Consolidation is airspace opacity without volume loss — fluid, pus, or cells replace "
        "air in alveoli. The model detects high-attenuation regions distinguishable from "
        "atelectasis by absence of volume loss signs."
    ),
    "Effusion": (
        "Pleural effusion is fluid in the pleural space, seen as dependent opacification "
        "with a meniscus sign. The model detects this as a smooth lower-zone opacity that "
        "blunts the costophrenic angle."
    ),
    "Edema": (
        "Pulmonary edema shows as bilateral perihilar haziness, Kerley B lines, and "
        "cardiomegaly. The model detects the bilateral symmetric opacity distribution "
        "and cardiac enlargement together."
    ),
    "Cardiomegaly": (
        "Cardiomegaly is an enlarged cardiac silhouette (cardiothoracic ratio > 0.5). "
        "The model keys on the width of the cardiac shadow relative to the thorax width."
    ),
}


def _generate_narrative(
    primary_label: str,
    primary_score: float,
    decision: str,
    raw_scores: dict[str, float],
    final_scores: dict[str, float],
    ruled_out: list[RuledOutHypothesis],
    pleural_log: list[str],
    conflict_log: list[str],
    pleural_feats_dict: Optional[dict],
) -> str:
    thr = get_threshold(primary_label)
    decision_str = _DECISION_PHRASES.get(decision, decision)
    class_desc = _CLASS_CLINICAL_DESC.get(primary_label, "")

    paragraphs: list[str] = []

    # 1. Decision summary
    paragraphs.append(
        f"The model classified this chest X-ray as **{primary_label}** "
        f"({primary_score:.1%} confidence) — a {decision_str}. "
        f"The class-specific positive threshold is {thr:.1%}; the final adjusted score "
        f"is {'above' if primary_score >= thr else 'below'} this threshold."
    )

    # 2. Clinical basis
    if class_desc:
        paragraphs.append(f"**Clinical basis:** {class_desc}")

    # 3. Pleural analysis contribution
    if pleural_feats_dict:
        subtype = pleural_feats_dict.get("subtype", "")
        ptx_ev = pleural_feats_dict.get("ptx_evidence", 0)
        emph_ev = pleural_feats_dict.get("emphysema_evidence", 0)
        bull_ev = pleural_feats_dict.get("bullous_evidence", 0)
        pl_line = pleural_feats_dict.get("pleural_line_score", 0)
        vasc = pleural_feats_dict.get("vascular_density", 0)
        hyper = pleural_feats_dict.get("hyperinflation_score", 0)
        focal = pleural_feats_dict.get("focal_avascular_score", 0)

        parts = [
            f"pleural-line confidence {pl_line:.1%}",
            f"vascular marking density {vasc:.1%}",
            f"hyperinflation index {hyper:.1%}",
            f"focal avascular score {focal:.1%}",
        ]
        paragraphs.append(
            f"**Image feature analysis:** The pleural/vascular analyzer reported "
            f"{', '.join(parts)}. Combined evidence scores — "
            f"Pneumothorax: {ptx_ev:.1%}, Emphysema: {emph_ev:.1%}, Bullous: {bull_ev:.1%}. "
            f"Best-fit image subtype: **{subtype}**."
        )

    # 4. Score corrections applied
    all_log = pleural_log + conflict_log
    if all_log:
        log_text = "; ".join(all_log)
        paragraphs.append(
            f"**Adjustments applied:** {len(all_log)} correction(s) were made to raw "
            f"DenseNet scores: {log_text}."
        )

    # 5. Competing hypotheses
    if ruled_out:
        ruled_text = ", ".join(
            f"{r.label} ({r.raw_score:.1%}→{r.final_score:.1%})"
            for r in ruled_out[:4]
        )
        paragraphs.append(
            f"**Competing hypotheses ruled out:** The following classes scored above the "
            f"review threshold on raw output but were suppressed by clinical conflict "
            f"penalties or image analysis: {ruled_text}."
        )

    # 6. Uncertainty note
    non_nf = [s for k, s in raw_scores.items() if k != "No Finding"]
    high_count = sum(1 for s in non_nf if 0.58 <= s <= 0.67)
    if high_count >= 5:
        paragraphs.append(
            f"**Uncertainty note:** {high_count} classes scored between 58–67% on "
            f"the raw DenseNet output simultaneously. This is a known model artefact "
            f"('score clustering') that indicates the model is uncertain. The post-hoc "
            f"conflict resolution and image analysis systems were essential in identifying "
            f"{primary_label} as the most plausible finding."
        )

    # 7. Limitations
    paragraphs.append(
        "**Limitations:** This output is generated by an AI model and must not be used as "
        "the sole basis for clinical decisions. The DenseNet was trained on NLP-labelled "
        "data (not direct expert annotation), operates at 224×224px resolution, and does "
        "not have access to clinical history, prior imaging, or laboratory values."
    )

    return "\n\n".join(paragraphs)


# ── Public API ────────────────────────────────────────────────────────────────

def build_explanation(
    primary_label: str,
    primary_score: float,
    decision: str,
    raw_scores: dict[str, float],
    final_scores: dict[str, float],
    pleural_log: list[str],
    conflict_log: list[str],
    pleural_feats_dict: Optional[dict] = None,
) -> ModelExplanation:
    """Build a full ModelExplanation from adapter outputs."""
    evidence = _build_evidence_chain(
        primary_label, raw_scores, pleural_log, conflict_log, primary_score
    )
    ruled_out = _build_ruled_out(
        primary_label, raw_scores, final_scores, conflict_log, pleural_log
    )
    biases = _select_biases(primary_label, raw_scores)
    narrative = _generate_narrative(
        primary_label, primary_score, decision,
        raw_scores, final_scores, ruled_out,
        pleural_log, conflict_log, pleural_feats_dict,
    )

    return ModelExplanation(
        primary_label=primary_label,
        primary_score=primary_score,
        decision=decision,
        evidence_chain=evidence,
        ruled_out=ruled_out,
        model_biases=biases,
        clinical_narrative=narrative,
    )
