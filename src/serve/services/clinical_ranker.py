"""Clinical plausibility engine for torchxrayvision outputs.

Applies conflict penalties to suppress pathophysiologically implausible
co-predictions (e.g., emphysema + fibrosis), then groups surviving findings
into multi-head clinical categories and produces a clinically ranked output.

Steps
-----
1. apply_conflict_rules(scores) → adjusted_scores
   Penalise targets whose driver class scores above trigger threshold.

2. build_clinical_groups(adjusted_scores, thresholds) → ClinicalOutput
   Group findings into Interstitial / Obstructive / Acute / Mass / Cardiac
   within each group, rank by adjusted score.

3. rank_primary(clinical_output) → top label + status
   Pick the highest-confidence finding across all groups as the primary label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config.class_conflicts import (
    CLASS_TO_GROUP,
    CLINICAL_GROUPS,
    CONFLICT_RULES,
    ConflictRule,
)
from src.config.txrv_thresholds import (
    REVIEW_MIN_SCORE,
    get_threshold,
)


# ── Step 1: conflict penalty ──────────────────────────────────────────────────

def apply_conflict_rules(
    scores: dict[str, float],
    rules: list[ConflictRule] | None = None,
) -> dict[str, float]:
    """Return adjusted scores with conflict penalties applied.

    Scores are clamped to [0, 1].  Original dict is not mutated.
    """
    if rules is None:
        rules = CONFLICT_RULES

    adj = dict(scores)
    applied: list[str] = []

    for rule in rules:
        driver_score = adj.get(rule.driver, 0.0)
        if driver_score < rule.driver_trigger:
            continue
        if rule.target not in adj:
            continue
        raw = adj[rule.target]
        penalised = max(0.0, raw - rule.penalty * driver_score)
        if penalised < raw:
            applied.append(
                f"{rule.target} {raw:.3f}→{penalised:.3f} "
                f"(driver={rule.driver} @ {driver_score:.3f})"
            )
            adj[rule.target] = penalised

    return adj, applied  # type: ignore[return-value]


# ── Step 2: clinical group building ──────────────────────────────────────────

@dataclass
class GroupFinding:
    name: str
    raw_score: float
    adjusted_score: float
    status: str    # "positive" | "review" | "suppressed"
    threshold: float
    penalised: bool


@dataclass
class ClinicalGroupResult:
    group_name: str
    findings: list[GroupFinding] = field(default_factory=list)

    @property
    def top_finding(self) -> GroupFinding | None:
        active = [f for f in self.findings if f.status in ("positive", "review")]
        return active[0] if active else None

    @property
    def has_positive(self) -> bool:
        return any(f.status == "positive" for f in self.findings)


@dataclass
class ClinicalOutput:
    groups: list[ClinicalGroupResult]
    adjusted_scores: dict[str, float]
    raw_scores: dict[str, float]
    conflict_log: list[str]

    def all_positives(self) -> list[GroupFinding]:
        out = []
        for g in self.groups:
            out.extend(f for f in g.findings if f.status == "positive")
        out.sort(key=lambda f: f.adjusted_score, reverse=True)
        return out

    def all_review(self) -> list[GroupFinding]:
        out = []
        for g in self.groups:
            out.extend(f for f in g.findings if f.status == "review")
        out.sort(key=lambda f: f.adjusted_score, reverse=True)
        return out

    def to_api_groups(self) -> dict[str, list[dict]]:
        """Serialisable dict for the API response."""
        result = {}
        for g in self.groups:
            active = [
                {
                    "name": f.name,
                    "score": round(f.adjusted_score, 4),
                    "raw_score": round(f.raw_score, 4),
                    "status": f.status,
                    "threshold": round(f.threshold, 4),
                    "penalised": f.penalised,
                }
                for f in g.findings
                if f.status in ("positive", "review")
            ]
            if active:
                result[g.group_name] = active
        return result


def build_clinical_output(
    raw_scores: dict[str, float],
    conflict_rules: list[ConflictRule] | None = None,
) -> ClinicalOutput:
    """Full pipeline: conflict penalties → group findings → ranked output."""
    adj_scores, conflict_log = apply_conflict_rules(raw_scores, conflict_rules)

    group_results: list[ClinicalGroupResult] = []

    for group_name, members in CLINICAL_GROUPS.items():
        findings: list[GroupFinding] = []
        for cls in members:
            raw = raw_scores.get(cls)
            if raw is None:
                continue
            adj = adj_scores.get(cls, raw)
            thr = get_threshold(cls)
            penalised = adj < raw - 0.001

            if adj >= thr:
                status = "positive"
            elif adj >= REVIEW_MIN_SCORE:
                status = "review"
            else:
                status = "suppressed"

            findings.append(GroupFinding(
                name=cls,
                raw_score=raw,
                adjusted_score=adj,
                status=status,
                threshold=thr,
                penalised=penalised,
            ))

        findings.sort(key=lambda f: f.adjusted_score, reverse=True)
        group_results.append(ClinicalGroupResult(group_name=group_name, findings=findings))

    return ClinicalOutput(
        groups=group_results,
        adjusted_scores=adj_scores,
        raw_scores=raw_scores,
        conflict_log=conflict_log,
    )
