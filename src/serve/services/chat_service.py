"""Medical report chat service.

Answers user questions about CXR analysis results.

Priority:
  1. OpenAI GPT (if OPENAI_API_KEY is set)
  2. Rule-based responder using analysis context (always available)
"""
from __future__ import annotations

import os
import re
from typing import Optional

from src.common.logging import get_logger

logger = get_logger("serve.chat")

# ── Medical knowledge base ────────────────────────────────────────────────────

_PATHOLOGY_EXPLAIN = {
    "Emphysema": (
        "Emphysema is a chronic obstructive pulmonary disease where the alveolar walls are "
        "permanently destroyed, leading to air trapping and hyperinflated lungs. On a chest "
        "X-ray it appears as flattened diaphragms, increased lung lucency, and reduced "
        "vascular markings. It is strongly associated with long-term smoking."
    ),
    "Pneumothorax": (
        "Pneumothorax is the presence of free air in the pleural space between the lung and "
        "chest wall. On CXR it shows as a peripheral avascular zone bounded by a sharp "
        "visceral pleural line. Tension pneumothorax is a medical emergency. Treatment "
        "depends on size: observation, aspiration, or chest drain."
    ),
    "Pneumonia": (
        "Pneumonia is an infection causing inflammation and consolidation in one or more lung "
        "lobes. CXR shows increased opacity (whitening) in the affected area, sometimes with "
        "air bronchograms. It can be bacterial, viral, or atypical. Treatment involves "
        "antibiotics for bacterial causes."
    ),
    "Fibrosis": (
        "Pulmonary fibrosis is scarring of lung tissue, typically appearing as reticular "
        "(net-like) opacities, often basal and peripheral, with volume loss. It can be "
        "idiopathic (IPF) or secondary to connective tissue diseases, drugs, or "
        "environmental exposures. It causes progressive breathlessness."
    ),
    "Atelectasis": (
        "Atelectasis is collapse of part or all of a lung. It appears as increased density "
        "with volume loss — the affected area looks whiter and smaller. It can be caused by "
        "mucus plugging, compression, or obstruction. Linear atelectasis appears as "
        "horizontal white lines (Fleischner lines)."
    ),
    "Consolidation": (
        "Consolidation is airspace opacity where alveoli fill with fluid, pus, or cells. "
        "Unlike atelectasis, there is no volume loss. It appears as a dense white area, "
        "often with an air bronchogram (dark airways visible through the opacity). "
        "Common causes include pneumonia, pulmonary oedema, and haemorrhage."
    ),
    "Effusion": (
        "Pleural effusion is fluid accumulation in the pleural space. On an erect CXR it "
        "appears as blunting of the costophrenic angle and a meniscus-shaped opacity at "
        "the lung base. Small effusions (<300ml) may only blunt the costophrenic angle. "
        "Causes include heart failure, malignancy, infection, and liver disease."
    ),
    "Edema": (
        "Pulmonary oedema is fluid accumulation in the lung interstitium and alveoli. "
        "It appears as bilateral perihilar haziness, Kerley B lines, and often with "
        "cardiomegaly. Cardiogenic oedema (heart failure) shows a 'bat wing' pattern. "
        "Treatment addresses the underlying cause — diuretics for cardiac oedema."
    ),
    "Cardiomegaly": (
        "Cardiomegaly is an enlarged heart, defined as cardiothoracic ratio > 0.5 on PA CXR. "
        "Causes include heart failure, cardiomyopathy, and valvular disease. On AP films "
        "the heart appears larger due to beam geometry — always verify with PA view."
    ),
    "Enlarged Cardiomediastinum": (
        "An enlarged cardiomediastinal silhouette can indicate cardiomegaly, pericardial "
        "effusion, mediastinal mass, or aortic aneurysm. Further imaging (echocardiogram "
        "or CT) is usually needed to characterise the cause."
    ),
    "Mass": (
        "A pulmonary mass (>3cm) appears as a rounded opacity. It raises concern for primary "
        "lung malignancy (especially in smokers), metastasis, or benign lesions like "
        "hamartoma. CT and biopsy are typically required for characterisation."
    ),
    "Nodule": (
        "A pulmonary nodule (<3cm) is a rounded opacity that requires follow-up based on "
        "risk factors (smoking, age, size, shape). The Fleischner Society guidelines "
        "recommend CT follow-up intervals based on nodule size and patient risk. "
        "Calcified nodules are generally benign."
    ),
    "Fracture": (
        "Rib or clavicular fractures appear as cortical discontinuities. They may be subtle "
        "on plain CXR. Associated pneumothorax, haemothorax, or pneumonia should be "
        "excluded. Multiple rib fractures (flail chest) are a surgical emergency."
    ),
    "Pleural Thickening": (
        "Pleural thickening is fibrosis of the pleural membrane, appearing as a white line "
        "along the chest wall. It may follow pleuritis, haemothorax, or asbestos exposure. "
        "Extensive bilateral pleural thickening (asbestosis) restricts breathing."
    ),
    "Infiltration": (
        "Infiltration describes non-specific increased lung density, which may represent "
        "infection, inflammation, aspiration, or early interstitial disease. It is a "
        "descriptive term that requires clinical correlation."
    ),
    "Lung Opacity": (
        "Lung opacity is a non-specific term for increased whiteness in the lung field. "
        "It encompasses consolidation, atelectasis, oedema, and haemorrhage. The pattern, "
        "distribution, and clinical context determine the likely cause."
    ),
    "No Finding": (
        "No significant radiographic abnormality was detected. The lung fields, cardiac "
        "silhouette, bony thorax, and mediastinum appear within normal limits for this study."
    ),
}

_NEXT_STEPS = {
    "Emphysema": "Pulmonary function tests (spirometry), smoking cessation counselling, COPD management.",
    "Pneumothorax": "Urgent clinical assessment. If >2cm or symptomatic: chest drain. If small/stable: observe.",
    "Pneumonia": "Assess CURB-65 severity. Oral antibiotics for community-acquired, IV for severe cases.",
    "Fibrosis": "HRCT chest, pulmonary function tests, rheumatology/respiratory referral.",
    "Atelectasis": "Physiotherapy, bronchoscopy if mucus plugging suspected, treat underlying cause.",
    "Consolidation": "Clinical correlation. If infective: antibiotics. Repeat CXR at 6 weeks to ensure resolution.",
    "Effusion": "Assess underlying cause (echo, LFTs, pleural tap if large). Drain if symptomatic.",
    "Edema": "Echocardiogram, BNP, diuretics if cardiogenic. Treat underlying cause.",
    "Cardiomegaly": "Echocardiogram to assess cardiac function and valves.",
    "Mass": "Urgent CT chest with contrast. Respiratory / oncology referral.",
    "Nodule": "CT chest and Fleischner Society follow-up protocol based on size and risk.",
    "No Finding": "No immediate action required. Correlate with clinical symptoms.",
}

# ── Rule-based responder ──────────────────────────────────────────────────────

def _rule_based_reply(question: str, ctx: dict) -> str:  # noqa: C901
    """Generate a structured reply from analysis context without an LLM."""
    q = question.lower().strip()

    label    = ctx.get("label", "Unknown")
    score    = ctx.get("calibrated_score", 0.0)
    decision = ctx.get("decision", "unknown")
    thr      = ctx.get("threshold", 0.5)
    expl     = ctx.get("model_explanation") or {}
    narrative    = expl.get("clinical_narrative", "")
    evidence_chain = expl.get("evidence_chain") or []
    biases       = expl.get("model_biases") or []
    ruled_out    = expl.get("ruled_out") or []
    positive     = ctx.get("positive_findings") or []
    review       = ctx.get("review_findings") or []
    conflict_log = ctx.get("conflict_log") or []

    # ── 1. Greeting / help ───────────────────────────────────────────────────
    if any(k in q for k in ["hello", "hi ", "help", "what can you", "capability"]):
        state = f"Current analysis: **{label}** ({score:.0%}, {decision.replace('_', ' ')})." if label != "Unknown" else ""
        return (
            f"{state}\n\nI can answer questions about this X-ray result. Try:\n\n"
            "• *Why did the model diagnose this?*\n"
            "• *What does emphysema mean?*\n"
            "• *What findings were suppressed and why?*\n"
            "• *What are the recommended next steps?*\n"
            "• *What biases affect this prediction?*\n"
            "• *How confident is the model?*"
        ).strip()

    # ── 2. Reasoning / evidence / why / how ──────────────────────────────────
    if any(k in q for k in ["why", "how did", "how does", "reason", "evidence",
                             "basis", "explain", "reasoning", "think", "decid"]):
        if evidence_chain:
            steps = "\n".join(
                f"• **{s['stage']}**: {float(s['value_before']):.0%} → "
                f"{float(s['value_after']):.0%}  _{s['reason']}_"
                for s in evidence_chain
            )
            return (
                f"**Why the model concluded {label} ({score:.0%}):**\n\n"
                f"{steps}\n\n"
                f"Class threshold: {thr:.0%} → decision: **{decision.replace('_', ' ')}**."
            )
        if narrative:
            return narrative
        return (
            f"The model scored **{label}** at **{score:.0%}** via DenseNet-121 applied "
            f"to a 224×224 grayscale input. The class-specific threshold is {thr:.0%}. "
            "No detailed evidence chain is available for this result."
        )

    # ── 3. Meaning / definition of the primary label or a named pathology ────
    if any(k in q for k in ["mean", "is ", "definition", "what is", "describe", "symptom", "look like"]):
        # Check if user named a specific class
        for cls, desc in _PATHOLOGY_EXPLAIN.items():
            if cls.lower() in q:
                steps = _NEXT_STEPS.get(cls, "Specialist referral recommended.")
                return f"**{cls}**\n\n{desc}\n\n**Next steps:** {steps}"
        # Fall back to primary label
        desc = _PATHOLOGY_EXPLAIN.get(label, "No clinical description available.")
        steps = _NEXT_STEPS.get(label, "Specialist referral recommended.")
        return f"**{label}**\n\n{desc}\n\n**Next steps:** {steps}"

    # ── 4. Specific pathology name in question → definition + steps ──────────
    for cls, desc in _PATHOLOGY_EXPLAIN.items():
        if cls.lower() in q:
            steps = _NEXT_STEPS.get(cls, "Specialist referral recommended.")
            return f"**{cls}**\n\n{desc}\n\n**Next steps:** {steps}"

    # ── 5. What did the model find / detect / results ────────────────────────
    if any(k in q for k in ["find", "detect", "result", "diagnos", "show", "report", "what did"]):
        pos_str = ", ".join(f"{f['name']} ({f['score']:.0%})" for f in positive) or "none above threshold"
        rev_str = ", ".join(f"{f['name']} ({f['score']:.0%})" for f in review) or "none"
        return (
            f"**Primary finding:** {label} — {score:.0%} ({decision.replace('_', ' ')}).\n\n"
            f"**Confirmed positives:** {pos_str}\n\n"
            f"**Under review:** {rev_str}\n\n"
            "Always correlate with clinical history and symptoms."
        )

    # ── 6. Suppressed / conflict / ruled out ─────────────────────────────────
    if any(k in q for k in ["suppress", "rule out", "ruled", "conflict", "penali", "remov", "not detect"]):
        parts = []
        if ruled_out:
            parts.append("**Hypotheses suppressed by conflict resolution:**")
            for r in ruled_out:
                parts.append(
                    f"• **{r['label']}** ({float(r['raw_score']):.0%} → "
                    f"{float(r['final_score']):.0%}): {r['suppressed_by']}"
                )
        if conflict_log:
            parts.append("\n**Conflict log:**")
            parts += [f"• {c}" for c in conflict_log]
        if parts:
            return "\n".join(parts)
        return "No conflict penalties were applied — all scores are as output by the DenseNet."

    # ── 7. Bias / reliability / limitations ──────────────────────────────────
    if any(k in q for k in ["bias", "limit", "trust", "reliab", "accurate", "wrong",
                             "error", "mistake", "confident about"]):
        if biases:
            high = [b for b in biases if b.get("relevance") == "high"]
            selected = (high or biases)[:4]
            parts = "\n".join(
                f"• **{b['name']}** [{b['relevance']}]: {b['impact_on_this_case']}"
                for b in selected
            )
            return (
                f"**Model biases affecting this prediction:**\n\n{parts}\n\n"
                "This output is AI-generated and must not replace clinical judgement."
            )
        return (
            "Known limitations: trained on NIH CXR14 (NLP-labelled, ~15-20% error rate), "
            "224×224px resolution, no access to clinical history or prior imaging."
        )

    # ── 8. Next steps / treatment / management ───────────────────────────────
    if any(k in q for k in ["next", "step", "do now", "treat", "manag", "action", "recommend", "should i"]):
        steps = _NEXT_STEPS.get(label, "Refer to a specialist for further workup.")
        return (
            f"**Recommended next steps for {label}:**\n\n{steps}\n\n"
            "⚠️ General guidance only — apply clinical judgement."
        )

    # ── 9. Score / confidence ────────────────────────────────────────────────
    if any(k in q for k in ["score", "prob", "percent", "%", "confident", "certain", "how sure"]):
        return (
            f"**{label}** final confidence: **{score:.1%}** "
            f"(class threshold {thr:.1%} → {decision.replace('_', ' ')}). "
            f"The raw DenseNet output was adjusted by pleural analysis and conflict rules "
            f"before this score was reached."
        )

    # ── 10. Normal / negative ────────────────────────────────────────────────
    if any(k in q for k in ["normal", "negative", "clear", "fine", "healthy"]):
        if decision in ("negative", "likely_normal_or_uncertain"):
            return (
                "The model found no findings above threshold. A normal AI result does not "
                "exclude pathology — always review the image and correlate clinically."
            )
        return (
            f"This study is **not normal** — primary finding is **{label}** "
            f"({score:.0%}, {decision.replace('_', ' ')}). Clinical review recommended."
        )

    # ── 11. Catch-all: return full narrative or summary ──────────────────────
    if narrative:
        return narrative

    desc = _PATHOLOGY_EXPLAIN.get(label, "")
    steps = _NEXT_STEPS.get(label, "Specialist referral recommended.")
    return (
        f"**Primary finding: {label}** ({score:.0%}, {decision.replace('_', ' ')}).\n\n"
        f"{desc}\n\n"
        f"**Next steps:** {steps}\n\n"
        "Ask me about the reasoning, suppressed findings, model biases, or confidence."
    )


# ── OpenAI responder ──────────────────────────────────────────────────────────

def _build_system_prompt(ctx: dict) -> str:
    label = ctx.get("label", "Unknown")
    score = ctx.get("calibrated_score", 0.0)
    decision = ctx.get("decision", "unknown")
    positive = ctx.get("positive_findings", [])
    review = ctx.get("review_findings", [])
    conflict_log = ctx.get("conflict_log", [])
    narrative = (ctx.get("model_explanation") or {}).get("clinical_narrative", "")

    pos_str = ", ".join(f"{f['name']} ({f['score']:.0%})" for f in positive) or "none"
    rev_str = ", ".join(f"{f['name']} ({f['score']:.0%})" for f in review) or "none"
    conflict_str = "; ".join(conflict_log[:5]) or "none"

    return f"""You are a medical AI assistant helping clinicians understand chest X-ray analysis results.
You are NOT a replacement for a qualified radiologist or clinician.
Always recommend clinical correlation and discourage sole reliance on AI output.

CURRENT ANALYSIS RESULT:
- Primary finding: {label} ({score:.0%} confidence) — {decision}
- Confirmed positives: {pos_str}
- Review findings: {rev_str}
- Conflict adjustments: {conflict_str}

CLINICAL NARRATIVE FROM MODEL:
{narrative}

INSTRUCTIONS:
- Answer questions about the above CXR analysis result
- Explain medical terms in plain English when asked
- Suggest appropriate next steps and investigations
- Always include a disclaimer about AI limitations
- Be concise but thorough; use bullet points where helpful
- Do not invent findings not present in the analysis above
"""


def _openai_reply(messages: list[dict], ctx: dict) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        system = _build_system_prompt(ctx)
        full_messages = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=full_messages,
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        logger.warning("OpenAI chat failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_reply(messages: list[dict], ctx: dict) -> tuple[str, str]:
    """Return (reply_text, model_used).

    Args:
        messages: list of {role, content} dicts (user/assistant turns)
        ctx:      PrimaryPrediction dict from the analysis result
    """
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Try OpenAI first
    openai_reply = _openai_reply(messages, ctx)
    if openai_reply:
        return openai_reply, os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Fallback: rule-based
    reply = _rule_based_reply(last_user, ctx)
    return reply, "rule-based"
