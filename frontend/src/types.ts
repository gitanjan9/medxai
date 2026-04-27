export interface Finding {
  name: string;
  score: number;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  x1_norm?: number;
  y1_norm?: number;
  x2_norm?: number;
  y2_norm?: number;
  width?: number;
  height?: number;
  area_fraction?: number;
}

export interface PrimaryPrediction {
  label: string;
  raw_score: number;
  calibrated_score: number;
  all_scores?: Record<string, number>;
  decision: "positive" | "review_required" | "review" | "negative" | "likely_normal_or_uncertain";
  confidence_band: "high" | "medium" | "low";
  review_reason?: string;
  threshold_version: string;
  threshold?: number;
  positive_findings?: Finding[];
  review_findings?: Finding[];
  clinical_groups?: Record<string, Array<{
    name: string;
    score: number;
    raw_score: number;
    status: string;
    threshold: number;
    penalised: boolean;
  }>>;
  conflict_log?: string[];
  model_explanation?: ModelExplanation;
}

export interface EvidenceStep {
  stage: string;
  value_before: number;
  value_after: number;
  delta: number;
  reason: string;
}

export interface RuledOutHypothesis {
  label: string;
  raw_score: number;
  final_score: number;
  suppressed_by: string;
  delta: number;
}

export interface ModelBias {
  name: string;
  relevance: "high" | "medium" | "low";
  description: string;
  impact_on_this_case: string;
}

export interface ModelExplanation {
  primary_label: string;
  primary_score: number;
  decision: string;
  evidence_chain: EvidenceStep[];
  ruled_out: RuledOutHypothesis[];
  model_biases: ModelBias[];
  clinical_narrative: string;
}

export interface OodSection {
  enabled: boolean;
  score?: number;
  decision?: "accept" | "review" | "reject";
  reason?: string;
}

export interface PathologySection {
  enabled: boolean;
  status?: string;
  top_finding?: string;
  top_score?: number;
  findings?: Finding[];
}

export interface LocalizationSection {
  enabled: boolean;
  type?: string;
  bbox?: BBox | null;
  region?: string | null;
  region_description?: string;
  disclaimer?: string;
}

export interface InferenceResponse {
  request_id: string;
  model_version: string;
  status: "ok" | "degraded";
  primary_prediction: PrimaryPrediction;
  ood: OodSection;
  pathologies: PathologySection;
  localization: LocalizationSection;
  warnings: string[];
  gradcam_base64?: string | null;
  gradcam_available?: boolean;
}

export interface AuditRow {
  id: string;
  timestamp: string;
  prediction: string;
  decision: "positive" | "review_required" | "review" | "negative" | "likely_normal_or_uncertain";
  confidence: number;
  request_id: string;
}
