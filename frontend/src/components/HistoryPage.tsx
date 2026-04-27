import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ClipboardList,
  CheckCircle2,
  AlertCircle,
  MinusCircle,
  HelpCircle,
  Search,
  Download,
  ThumbsUp,
  ThumbsDown,
  Clock,
  RefreshCw,
} from "lucide-react";
import type { AuditRow } from "../types";
import { recordsApi, type PredictionRecord } from "../services/api";
import RecordDetailDrawer from "./RecordDetailDrawer";

interface Props {
  rows: AuditRow[];
  onBack: () => void;
}

type DecisionFilter = "all" | AuditRow["decision"];

const DECISION_META: Record<string, { label: string; icon: JSX.Element; pill: string; bar: string }> = {
  positive: {
    label: "Positive",
    icon: <CheckCircle2 className="w-4 h-4 text-blue-500" />,
    pill: "bg-blue-50 text-blue-700 border-blue-200",
    bar: "bg-blue-400",
  },
  review_required: {
    label: "Review Required",
    icon: <AlertCircle className="w-4 h-4 text-amber-500" />,
    pill: "bg-amber-50 text-amber-700 border-amber-200",
    bar: "bg-amber-400",
  },
  review: {
    label: "Review",
    icon: <AlertCircle className="w-4 h-4 text-amber-500" />,
    pill: "bg-amber-50 text-amber-700 border-amber-200",
    bar: "bg-amber-400",
  },
  likely_normal_or_uncertain: {
    label: "Uncertain",
    icon: <HelpCircle className="w-4 h-4 text-slate-400" />,
    pill: "bg-slate-100 text-slate-500 border-slate-200",
    bar: "bg-slate-300",
  },
  negative: {
    label: "Negative",
    icon: <MinusCircle className="w-4 h-4 text-emerald-500" />,
    pill: "bg-emerald-50 text-emerald-700 border-emerald-200",
    bar: "bg-emerald-400",
  },
};

function DecisionBadge({ decision }: { decision: AuditRow["decision"] }) {
  const meta = DECISION_META[decision] ?? DECISION_META.negative;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full border ${meta.pill}`}>
      {meta.label}
    </span>
  );
}

function StatCard({ label, value, sub, color }: { label: string; value: number; sub: string; color: string }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-card">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-0.5">{sub}</p>
    </div>
  );
}

function exportCSV(rows: AuditRow[]) {
  const header = ["Timestamp", "Prediction", "Decision", "Confidence (%)", "Request ID"];
  const lines = rows.map((r) => [
    r.timestamp,
    r.prediction,
    r.decision,
    (r.confidence * 100).toFixed(1),
    r.request_id,
  ]);
  const csv = [header, ...lines].map((row) => row.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `medicalxai_history_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function FeedbackBadge({ feedback }: { feedback: PredictionRecord["feedback"] }) {
  if (feedback === "correct") return (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-green-50 text-green-700 border border-green-200">
      <ThumbsUp className="w-3 h-3" /> Correct
    </span>
  );
  if (feedback === "wrong") return (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
      <ThumbsDown className="w-3 h-3" /> Corrected
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-400">
      <Clock className="w-3 h-3" /> Pending
    </span>
  );
}

export default function HistoryPage({ rows, onBack }: Props) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<DecisionFilter>("all");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [dbRecords, setDbRecords] = useState<PredictionRecord[]>([]);
  const [loadingDb, setLoadingDb] = useState(true);
  const [selectedRecord, setSelectedRecord] = useState<PredictionRecord | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleRowClick = useCallback((record: PredictionRecord) => {
    setSelectedRecord(record);
    setDrawerOpen(true);
  }, []);

  const handleFeedbackDone = useCallback((id: string, feedback: "correct" | "wrong") => {
    setDbRecords((prev) =>
      prev.map((r) => r.id === id ? { ...r, feedback } : r)
    );
    setSelectedRecord((prev) =>
      prev?.id === id ? { ...prev, feedback } : prev
    );
  }, []);

  const handlePatientSaved = useCallback((id: string, patch: Partial<PredictionRecord>) => {
    setDbRecords((prev) =>
      prev.map((r) => r.id === id ? { ...r, ...patch } : r)
    );
    setSelectedRecord((prev) =>
      prev?.id === id ? { ...prev, ...patch } : prev
    );
  }, []);

  useEffect(() => {
    recordsApi.list({ limit: 200 })
      .then(setDbRecords)
      .catch(() => {})
      .finally(() => setLoadingDb(false));
  }, []);

  // Merge DB records into display rows (DB is source of truth; session rows fill gaps)
  const dbIds = new Set(dbRecords.map((r) => r.request_id));
  const sessionOnlyRows = rows.filter((r) => !dbIds.has(r.request_id));

  const allRows: (AuditRow & { dbRecord?: PredictionRecord })[] = [
    ...dbRecords.map((r) => ({
      id: r.id,
      request_id: r.request_id,
      timestamp: new Date(r.created_at).toLocaleString("en-GB", {
        year: "numeric", month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit",
      }),
      prediction: r.primary_label,
      decision: r.decision as AuditRow["decision"],
      confidence: r.confidence,
      dbRecord: r,
    })),
    ...sessionOnlyRows.map((r) => ({ ...r, dbRecord: undefined })),
  ];

  const positives = allRows.filter((r) => r.decision === "positive").length;
  const reviews   = allRows.filter((r) => r.decision === "review_required" || r.decision === "review").length;
  const negatives = allRows.filter((r) => r.decision === "negative").length;
  const avgConf   = allRows.length ? allRows.reduce((s, r) => s + r.confidence, 0) / allRows.length : 0;

  const filtered = allRows
    .filter((r) => filter === "all" || r.decision === filter)
    .filter((r) =>
      search === "" ||
      r.prediction.toLowerCase().includes(search.toLowerCase()) ||
      r.decision.toLowerCase().includes(search.toLowerCase()) ||
      r.request_id.toLowerCase().includes(search.toLowerCase()) ||
      (r.dbRecord?.patient_name ?? "").toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) =>
      sortDir === "desc"
        ? b.timestamp.localeCompare(a.timestamp)
        : a.timestamp.localeCompare(b.timestamp)
    );

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-screen-xl mx-auto px-6 py-6"
    >
      {/* Page header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Workspace
        </button>
        <div className="h-5 w-px bg-slate-200" />
        <div className="flex items-center gap-2">
          <ClipboardList className="w-5 h-5 text-slate-400" />
          <h1 className="text-lg font-bold text-slate-800">Patient Report History</h1>
        </div>
        <span className="ml-auto text-sm text-slate-400">{rows.length} total studies</span>
        {rows.length > 0 && (
          <button
            onClick={() => exportCSV(rows)}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-600 border border-slate-200 bg-white hover:bg-slate-50 px-3 py-1.5 rounded-lg transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <StatCard label="Total Studies"   value={rows.length} sub="all time"          color="text-slate-800" />
        <StatCard label="Positives"       value={positives}   sub="confirmed findings" color="text-blue-600"  />
        <StatCard label="For Review"      value={reviews}     sub="require follow-up"  color="text-amber-600" />
        <StatCard label="Avg Confidence"  value={Math.round(avgConf * 100)} sub="% across all studies" color="text-emerald-600" />
      </div>

      {/* Filters + search */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="relative flex-1 min-w-48">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search prediction, decision, ID…"
            className="w-full text-sm pl-8 pr-3 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
          />
        </div>

        <div className="flex items-center gap-1.5 flex-wrap">
          {(["all", "positive", "review_required", "negative", "likely_normal_or_uncertain"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setFilter(d)}
              className={`text-xs font-medium px-3 py-1.5 rounded-full border transition-colors ${
                filter === d
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-500 border-slate-200 hover:bg-slate-50"
              }`}
            >
              {d === "all" ? "All" : DECISION_META[d]?.label ?? d}
            </button>
          ))}
        </div>

        <button
          onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
          className="text-xs font-medium text-slate-500 border border-slate-200 bg-white hover:bg-slate-50 px-3 py-1.5 rounded-lg transition-colors"
        >
          {sortDir === "desc" ? "↓ Newest first" : "↑ Oldest first"}
        </button>
      </div>

      {/* Table */}
      {loadingDb ? (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-card flex items-center justify-center py-16 gap-3 text-slate-400">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading records…</span>
        </div>
      ) : allRows.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-card flex flex-col items-center justify-center py-20 gap-3">
          <ClipboardList className="w-12 h-12 text-slate-200" />
          <p className="text-sm font-semibold text-slate-400">No studies yet</p>
          <p className="text-xs text-slate-400">Run an analysis from the workspace to see history here.</p>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50">
                  {["#", "Timestamp", "Patient", "Primary Finding", "Decision", "Confidence", "Feedback"].map((h) => (
                    <th
                      key={h}
                      className="px-5 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {filtered.map((row, i) => {
                  const meta = DECISION_META[row.decision] ?? DECISION_META.negative;
                  return (
                    <tr
                      key={row.id}
                      onClick={() => row.dbRecord && handleRowClick(row.dbRecord)}
                      className={`transition-colors ${
                        row.dbRecord
                          ? "hover:bg-blue-50 cursor-pointer"
                          : "hover:bg-slate-50"
                      }`}
                    >
                      <td className="px-5 py-3.5 text-xs text-slate-300 tabular-nums">
                        {i + 1}
                      </td>
                      <td className="px-5 py-3.5 text-xs text-slate-500 font-mono whitespace-nowrap">
                        {row.timestamp}
                      </td>
                      <td className="px-5 py-3.5 text-xs text-slate-500">
                        {row.dbRecord?.patient_name || <span className="text-slate-300">—</span>}
                      </td>
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-2">
                          {meta.icon}
                          <span className="text-sm font-semibold text-slate-800">
                            {row.prediction.replace(/_/g, " ")}
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <DecisionBadge decision={row.decision} />
                      </td>
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-2">
                          <div className="w-24 bg-slate-100 rounded-full h-2">
                            <div
                              className={`${meta.bar} h-2 rounded-full transition-all`}
                              style={{ width: `${row.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-semibold text-slate-700 tabular-nums w-9">
                            {(row.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        {row.dbRecord
                          ? <FeedbackBadge feedback={row.dbRecord.feedback} />
                          : <span className="text-xs text-slate-300">—</span>}
                        {row.dbRecord?.true_label && (
                          <p className="text-xs text-slate-400 mt-0.5">{row.dbRecord.true_label}</p>
                        )}
                      </td>
                      <td className="px-3 py-3.5 text-slate-300">
                        {row.dbRecord && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {filtered.length === 0 && (
            <div className="py-10 text-center text-sm text-slate-400">
              No studies match the current filter.
            </div>
          )}

          <div className="px-5 py-3 border-t border-slate-100 bg-slate-50 flex items-center justify-between">
            <p className="text-xs text-slate-400">
              Showing {filtered.length} of {allRows.length} studies
            </p>
            <p className="text-xs text-slate-400">
              {negatives} negative · {positives} positive · {reviews} for review
            </p>
          </div>
        </div>
      )}
      <RecordDetailDrawer
        record={selectedRecord}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onFeedbackDone={handleFeedbackDone}
        onPatientSaved={handlePatientSaved}
      />
    </motion.div>
  );
}
