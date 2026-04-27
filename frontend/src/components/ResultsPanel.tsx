import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  AlertCircle,
  MinusCircle,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Layers,
  MapPin,
  AlertTriangle,
  Info,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import type { InferenceResponse, EvidenceStep, RuledOutHypothesis, ModelBias } from "../types";
import FeedbackModal from "./FeedbackModal";
import { recordsApi } from "../services/api";

interface Props {
  result: InferenceResponse;
  predictionId?: string;   // DB record id returned by /v1/predict (if saved)
  initialFeedback?: "correct" | "wrong"; // pre-existing feedback from DB
}

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-white border border-slate-200 rounded-2xl shadow-card p-4 ${className}`}>
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
        {title}
      </p>
      {children}
    </div>
  );
}

function ScoreBar({ label, score }: { label: string; score: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-600 w-36 truncate">{label}</span>
      <div className="flex-1 bg-slate-100 rounded-full h-1.5">
        <div
          className="bg-blue-500 h-1.5 rounded-full transition-all duration-700"
          style={{ width: `${Math.min(score * 100, 100)}%` }}
        />
      </div>
      <span className="text-xs font-semibold text-slate-700 w-10 text-right tabular-nums">
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function DecisionPill({ decision }: { decision: string }) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    positive: {
      cls: "bg-red-50 text-red-700 border-red-200",
      icon: <AlertCircle className="w-3.5 h-3.5" />,
      label: "Positive",
    },
    review_required: {
      cls: "bg-amber-50 text-amber-700 border-amber-200",
      icon: <AlertTriangle className="w-3.5 h-3.5" />,
      label: "Review Required",
    },
    review: {
      cls: "bg-amber-50 text-amber-700 border-amber-200",
      icon: <AlertCircle className="w-3.5 h-3.5" />,
      label: "Needs Review",
    },
    likely_normal_or_uncertain: {
      cls: "bg-emerald-50 text-emerald-700 border-emerald-200",
      icon: <CheckCircle2 className="w-3.5 h-3.5" />,
      label: "Likely Normal / Uncertain",
    },
    negative: {
      cls: "bg-slate-100 text-slate-600 border-slate-200",
      icon: <MinusCircle className="w-3.5 h-3.5" />,
      label: "Negative",
    },
  };
  const { cls, icon, label } = map[decision] ?? map.review;
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${cls}`}>
      {icon}
      {label}
    </span>
  );
}

function labelDisplay(label: string) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ── Primary Prediction ─────────────────────────────────────────── */
function PredictionCard({ result, predictionId, initialFeedback }: Props) {
  const pp = result.primary_prediction;
  const score = pp.calibrated_score;
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackDone, setFeedbackDone] = useState<"correct" | "wrong" | null>(initialFeedback ?? null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  async function markCorrect() {
    if (!predictionId || feedbackLoading) return;
    setFeedbackLoading(true);
    try {
      await recordsApi.submitFeedback(predictionId, "correct");
      setFeedbackDone("correct");
    } catch {
      // silently ignore — still mark locally so UX isn't blocked
      setFeedbackDone("correct");
    } finally {
      setFeedbackLoading(false);
    }
  }

  return (
    <Card title="Primary Prediction">
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-lg font-bold text-slate-900 leading-tight">
            {labelDisplay(pp.label)}
          </p>
          <p className="text-xs text-slate-400 mt-0.5 font-mono">{pp.label}</p>
        </div>
        <DecisionPill decision={pp.decision} />
      </div>

      {/* Calibrated confidence bar */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs text-slate-500">Calibrated confidence</span>
          <span className="text-sm font-bold text-slate-800 tabular-nums">
            {(score * 100).toFixed(1)}%
            {pp.threshold != null && (
              <span className="ml-1 text-xs font-normal text-slate-400">
                / {(pp.threshold * 100).toFixed(0)}% threshold
              </span>
            )}
          </span>
        </div>
        <div className="bg-slate-100 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-700 ${
              pp.decision === "positive"
                ? "bg-red-500"
                : pp.decision === "review_required"
                ? "bg-amber-400"
                : pp.decision === "likely_normal_or_uncertain"
                ? "bg-emerald-400"
                : "bg-slate-300"
            }`}
            style={{ width: `${score * 100}%` }}
          />
        </div>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap gap-2 mb-3">
        <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded-lg">
          Band: <span className="text-slate-700">{pp.confidence_band}</span>
        </span>
        <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded-lg">
          Raw: <span className="text-slate-700">{(pp.raw_score * 100).toFixed(1)}%</span>
        </span>
        <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded-lg">
          Threshold: <span className="text-slate-700">{pp.threshold_version}</span>
        </span>
      </div>

      {/* Clinical disclaimer */}
      <div className="flex items-start gap-2 mb-3 px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl">
        <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500">
          AI finding is not a medical diagnosis.
          Scores below class threshold should not be treated as confirmed pathology.
        </p>
      </div>

      {/* Clinician feedback */}
      {predictionId && (
        feedbackDone ? (
          <div className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium ${
            feedbackDone === "correct"
              ? "bg-green-50 text-green-700 border border-green-200"
              : "bg-amber-50 text-amber-700 border border-amber-200"
          }`}>
            {feedbackDone === "correct"
              ? <><ThumbsUp className="w-3.5 h-3.5" /> Marked as correct</>  
              : <><ThumbsDown className="w-3.5 h-3.5" /> Correction submitted</>}
          </div>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={markCorrect}
              disabled={feedbackLoading}
              className="flex-1 flex items-center justify-center gap-1.5 text-xs font-medium py-2 rounded-xl border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 disabled:opacity-60 transition-colors"
            >
              {feedbackLoading
                ? <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v4m0 8v4M4 12H8m8 0h4" /></svg>
                : <ThumbsUp className="w-3.5 h-3.5" />}
              Correct
            </button>
            <button
              onClick={() => setFeedbackOpen(true)}
              className="flex-1 flex items-center justify-center gap-1.5 text-xs font-medium py-2 rounded-xl border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
            >
              <ThumbsDown className="w-3.5 h-3.5" /> Wrong
            </button>
          </div>
        )
      )}

      <FeedbackModal
        predictionId={predictionId ?? ""}
        predictedLabel={labelDisplay(pp.label)}
        open={feedbackOpen}
        onClose={() => setFeedbackOpen(false)}
        onDone={(fb) => { setFeedbackOpen(false); setFeedbackDone(fb); }}
      />

      {/* Confirmed positive findings */}
      {pp.positive_findings && pp.positive_findings.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-red-600 mb-1.5">Confirmed findings</p>
          <div className="space-y-1">
            {pp.positive_findings.map((f) => (
              <div key={f.name} className="flex items-center justify-between px-2.5 py-1.5 bg-red-50 border border-red-100 rounded-lg">
                <span className="text-xs font-medium text-red-800">{f.name}</span>
                <span className="text-xs font-bold text-red-700 tabular-nums">{(f.score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Review findings */}
      {pp.review_findings && pp.review_findings.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-amber-600 mb-1.5">Needs review (below threshold)</p>
          <div className="space-y-1">
            {pp.review_findings.map((f) => (
              <div key={f.name} className="flex items-center justify-between px-2.5 py-1.5 bg-amber-50 border border-amber-100 rounded-lg">
                <span className="text-xs font-medium text-amber-800">{f.name}</span>
                <span className="text-xs font-bold text-amber-700 tabular-nums">{(f.score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {pp.review_reason && (
        <p className="text-xs text-amber-600 bg-amber-50 border border-amber-100 rounded-lg px-3 py-1.5">
          {pp.review_reason}
        </p>
      )}

      {/* All scores */}
      {pp.all_scores && (
        <div className="mt-3 space-y-1.5">
          <p className="text-xs font-medium text-slate-400 mb-2">All class scores</p>
          {Object.entries(pp.all_scores)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, sc]) => (
              <ScoreBar key={cls} label={labelDisplay(cls)} score={sc} />
            ))}
        </div>
      )}

      {/* Request ID */}
      <div className="mt-3 pt-3 border-t border-slate-100">
        <p className="text-xs text-slate-400 font-mono truncate">
          ID: {result.request_id}
        </p>
      </div>
    </Card>
  );
}

/* ── Clinical Groups Card ───────────────────────────────────────── */
const GROUP_COLOURS: Record<string, { bg: string; border: string; title: string; dot: string }> = {
  "Interstitial / Restrictive": { bg: "bg-purple-50",  border: "border-purple-100", title: "text-purple-700", dot: "bg-purple-400" },
  "Obstructive / Structural":   { bg: "bg-blue-50",    border: "border-blue-100",   title: "text-blue-700",   dot: "bg-blue-400" },
  "Acute Opacity / Consolidation": { bg: "bg-red-50",  border: "border-red-100",    title: "text-red-700",    dot: "bg-red-400" },
  "Mass / Nodular":             { bg: "bg-orange-50",  border: "border-orange-100", title: "text-orange-700", dot: "bg-orange-400" },
  "Cardiac":                    { bg: "bg-pink-50",    border: "border-pink-100",   title: "text-pink-700",   dot: "bg-pink-400" },
};

function ClinicalGroupsCard({ result }: Props) {
  const pp = result.primary_prediction;
  if (!pp.clinical_groups || Object.keys(pp.clinical_groups).length === 0) return null;

  return (
    <Card title="Multi-Head Clinical Analysis">
      <div className="space-y-3">
        {Object.entries(pp.clinical_groups).map(([group, findings]) => {
          const col = GROUP_COLOURS[group] ?? { bg: "bg-slate-50", border: "border-slate-100", title: "text-slate-700", dot: "bg-slate-400" };
          return (
            <div key={group} className={`rounded-xl border ${col.border} ${col.bg} p-3`}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-2 h-2 rounded-full flex-shrink-0 ${col.dot}`} />
                <p className={`text-xs font-semibold ${col.title}`}>{group}</p>
              </div>
              <div className="space-y-1">
                {findings.map((f) => (
                  <div key={f.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-slate-700">{f.name}</span>
                      {f.penalised && (
                        <span className="text-xs text-slate-400 italic">(adjusted)</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5">
                      {f.status === "positive" && (
                        <span className="text-xs font-semibold px-1.5 py-0.5 bg-red-100 text-red-700 rounded-full">+</span>
                      )}
                      <span className="text-xs font-semibold tabular-nums text-slate-700">
                        {(f.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      {pp.conflict_log && pp.conflict_log.length > 0 && (
        <details className="mt-3">
          <summary className="text-xs text-slate-400 cursor-pointer select-none">
            {pp.conflict_log.length} conflict adjustment{pp.conflict_log.length > 1 ? "s" : ""} applied
          </summary>
          <div className="mt-1.5 space-y-1">
            {pp.conflict_log.map((log, i) => (
              <p key={i} className="text-xs font-mono text-slate-400 bg-slate-50 px-2 py-1 rounded">
                {log}
              </p>
            ))}
          </div>
        </details>
      )}
    </Card>
  );
}

/* ── Explanation Panel ──────────────────────────────────────────── */
const BIAS_COLOUR = {
  high:   { pill: "bg-red-100 text-red-700",    dot: "bg-red-400"    },
  medium: { pill: "bg-amber-100 text-amber-700", dot: "bg-amber-400" },
  low:    { pill: "bg-slate-100 text-slate-500", dot: "bg-slate-300" },
};

function EvidenceScoreBar({ before, after, label }: { before: number; after: number; label: string }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const width = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="tabular-nums font-mono">{pct(before)} → {pct(after)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-slate-100 overflow-hidden">
        <div className="absolute inset-y-0 left-0 bg-slate-300 rounded-full" style={{ width: width(before) }} />
        <div className="absolute inset-y-0 left-0 bg-blue-400 rounded-full transition-all" style={{ width: width(after) }} />
      </div>
    </div>
  );
}

export function ExplanationPanel({ result }: Props) {
  const exp = result.primary_prediction.model_explanation;
  if (!exp) return null;

  const [open, setOpen] = React.useState(false);

  return (
    <Card title="Model Explanation & Bias Disclosure">
      {/* Narrative */}
      <div className="space-y-2 mb-4">
        {exp.clinical_narrative.split("\n\n").map((para, i) => {
          const parts = para.split(/\*\*(.*?)\*\*/g);
          return (
            <p key={i} className="text-xs text-slate-600 leading-relaxed">
              {parts.map((p, j) =>
                j % 2 === 1 ? <strong key={j} className="text-slate-800">{p}</strong> : p
              )}
            </p>
          );
        })}
      </div>

      {/* Evidence chain */}
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between text-xs font-semibold text-slate-600 bg-slate-50 hover:bg-slate-100 rounded-lg px-3 py-2 mb-1 transition-colors"
      >
        <span>Score evidence chain ({exp.evidence_chain.length} stages)</span>
        <span className="text-slate-400">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="space-y-3 mb-4 px-1">
          {exp.evidence_chain.map((step: EvidenceStep, i: number) => (
            <div key={i} className="border-l-2 border-blue-200 pl-3 space-y-1">
              <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide">{step.stage}</p>
              <EvidenceScoreBar before={step.value_before} after={step.value_after} label={step.stage} />
              <p className="text-xs text-slate-500 leading-relaxed">{step.reason}</p>
            </div>
          ))}
        </div>
      )}

      {/* Ruled-out */}
      {exp.ruled_out.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-semibold text-slate-600 mb-2">Competing hypotheses suppressed</p>
          <div className="space-y-2">
            {exp.ruled_out.map((r: RuledOutHypothesis) => (
              <div key={r.label} className="flex items-start gap-2 bg-slate-50 rounded-lg px-3 py-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className="text-xs font-semibold text-slate-700">{r.label}</span>
                    <span className="text-xs text-slate-400 tabular-nums">
                      {(r.raw_score * 100).toFixed(1)}% → {(r.final_score * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs font-semibold text-red-500 tabular-nums">
                      ({(r.delta * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-snug truncate">{r.suppressed_by}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bias cards */}
      <div>
        <p className="text-xs font-semibold text-slate-600 mb-2">Known model biases</p>
        <div className="space-y-2">
          {exp.model_biases.map((b: ModelBias) => {
            const col = BIAS_COLOUR[b.relevance];
            return (
              <details key={b.name} className="group rounded-lg border border-slate-100 bg-white overflow-hidden">
                <summary className="flex items-center gap-2 px-3 py-2 cursor-pointer select-none list-none">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${col.dot}`} />
                  <span className="flex-1 text-xs font-medium text-slate-700">{b.name}</span>
                  <span className={`text-xs font-semibold px-1.5 py-0.5 rounded-full ${col.pill}`}>
                    {b.relevance}
                  </span>
                </summary>
                <div className="px-3 pb-3 pt-1 border-t border-slate-50 space-y-1.5">
                  <p className="text-xs text-slate-500 leading-relaxed">{b.description}</p>
                  <p className="text-xs font-medium text-slate-600 leading-relaxed">
                    <span className="text-slate-400">Impact on this case: </span>{b.impact_on_this_case}
                  </p>
                </div>
              </details>
            );
          })}
        </div>
      </div>
    </Card>
  );
}

/* ── OOD Card ───────────────────────────────────────────────────── */
function OodCard({ result }: Props) {
  const ood = result.ood;
  if (!ood.enabled) return null;

  const icons = {
    accept: <ShieldCheck className="w-5 h-5 text-emerald-500" />,
    review: <ShieldAlert className="w-5 h-5 text-amber-500" />,
    reject: <ShieldX className="w-5 h-5 text-red-500" />,
  };
  const labels = {
    accept: { cls: "text-emerald-700 bg-emerald-50 border-emerald-200", txt: "Accepted as CXR" },
    review: { cls: "text-amber-700 bg-amber-50 border-amber-200", txt: "Needs Review" },
    reject: { cls: "text-red-700 bg-red-50 border-red-200", txt: "Rejected" },
  };
  const d = ood.decision ?? "review";
  const meta = labels[d as keyof typeof labels] ?? labels.review;

  return (
    <Card title="Image Quality Check">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icons[d as keyof typeof icons] ?? icons.review}
          <span className="text-sm font-semibold text-slate-800">OOD Detector</span>
        </div>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${meta.cls}`}>
          {meta.txt}
        </span>
      </div>
      {ood.score != null && (
        <ScoreBar label="CXR likelihood score" score={ood.score} />
      )}
      {ood.reason && (
        <p className="text-xs text-slate-500 mt-2">
          Reason: <span className="text-slate-700">{ood.reason}</span>
        </p>
      )}
    </Card>
  );
}

/* ── Pathology Card ─────────────────────────────────────────────── */
function PathologyCard({ result }: Props) {
  const p = result.pathologies;
  if (!p.enabled) return null;

  return (
    <Card title="Auxiliary Pathology Findings">
      <div className="flex items-center gap-2 mb-3 px-2.5 py-1.5 bg-blue-50 border border-blue-100 rounded-xl">
        <Layers className="w-3.5 h-3.5 text-blue-500 flex-shrink-0" />
        <p className="text-xs text-blue-700">
          Separate 18-class pathology model — not the primary classifier
        </p>
      </div>

      {p.status === "error" ? (
        <p className="text-xs text-red-600">Pathology model returned an error.</p>
      ) : (
        <>
          {p.top_finding && (
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-slate-500">Top finding</span>
              <span className="text-sm font-semibold text-slate-800">{p.top_finding}</span>
            </div>
          )}
          <div className="space-y-1.5">
            {(p.findings ?? []).slice(0, 8).map((f) => (
              <ScoreBar key={f.name} label={f.name} score={f.score} />
            ))}
          </div>
        </>
      )}
    </Card>
  );
}

/* ── Localization Card ──────────────────────────────────────────── */
function LocalizationCard({ result }: Props) {
  const loc = result.localization;
  if (!loc.enabled || loc.type !== "approximate_attention_region") return null;

  return (
    <Card title="Attention Localisation">
      <div className="flex items-start gap-2 mb-3 px-2.5 py-1.5 bg-amber-50 border border-amber-100 rounded-xl">
        <AlertTriangle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-amber-700">{loc.disclaimer}</p>
      </div>

      {loc.region && (
        <div className="flex items-center gap-2 mb-2">
          <MapPin className="w-3.5 h-3.5 text-blue-500" />
          <span className="text-sm font-semibold text-slate-800">
            {loc.region.replace(/_/g, " ")}
          </span>
        </div>
      )}

      {loc.region_description && (
        <p className="text-xs text-slate-500 mb-2">{loc.region_description}</p>
      )}

      {loc.bbox && (
        <div className="mt-2 grid grid-cols-2 gap-1.5 text-xs font-mono text-slate-500 bg-slate-50 rounded-xl p-2">
          <span>x1: {loc.bbox.x1}</span>
          <span>y1: {loc.bbox.y1}</span>
          <span>x2: {loc.bbox.x2}</span>
          <span>y2: {loc.bbox.y2}</span>
          {loc.bbox.area_fraction != null && (
            <span className="col-span-2">
              area: {(loc.bbox.area_fraction * 100).toFixed(1)}%
            </span>
          )}
        </div>
      )}
    </Card>
  );
}

/* ── Warnings Card ──────────────────────────────────────────────── */
function WarningsCard({ result }: Props) {
  const warnings = result.warnings ?? [];
  if (warnings.length === 0) return null;

  const labels: Record<string, string> = {
    approximate_localization_only:
      "Localisation is approximate only — not a validated lesion boundary.",
    ood_in_review_band:
      "Image quality is uncertain. OOD score is in the review band.",
    ood_check_failed: "OOD detector failed. Image authenticity not verified.",
    pathology_inference_failed:
      "Auxiliary pathology model failed. Primary prediction is unaffected.",
    localization_generation_failed: "Localisation could not be generated for this image.",
  };

  return (
    <Card title="Clinical Cautions">
      <div className="space-y-2">
        {warnings.map((w) => (
          <div
            key={w}
            className="flex items-start gap-2 px-3 py-2 bg-amber-50 border border-amber-100 rounded-xl"
          >
            <Info className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-800">
              {labels[w] ?? w.replace(/_/g, " ")}
            </p>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ── Combined panel ─────────────────────────────────────────────── */
export default function ResultsPanel({ result, predictionId, initialFeedback }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="flex flex-col gap-4"
    >
      <PredictionCard key={predictionId ?? "noid"} result={result} predictionId={predictionId} initialFeedback={initialFeedback} />
      <ClinicalGroupsCard result={result} />
      <OodCard result={result} />
      <PathologyCard result={result} />
      <LocalizationCard result={result} />
      <WarningsCard result={result} />
    </motion.div>
  );
}
