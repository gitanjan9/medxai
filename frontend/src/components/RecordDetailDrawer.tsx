import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Pencil, Check, Loader2, RefreshCw, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import { recordsApi, type PredictionRecord } from "../services/api";
import type { InferenceResponse } from "../types";
import ResultsPanel from "./ResultsPanel";
import FeedbackModal from "./FeedbackModal";

interface Props {
  record: PredictionRecord | null;
  open: boolean;
  onClose: () => void;
  onFeedbackDone: (id: string, feedback: "correct" | "wrong") => void;
  onPatientSaved?: (id: string, patch: Partial<PredictionRecord>) => void;
}

const GENDER_OPTIONS = ["Male", "Female", "Non-binary", "Prefer not to say"];

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="text-slate-400 w-20 flex-shrink-0 text-xs pt-0.5">{label}</span>
      <span className="text-slate-700 font-medium text-sm">{value}</span>
    </div>
  );
}

function Field({
  label, value, onChange, type = "text", placeholder = "", options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  options?: string[];
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-500 mb-1">{label}</label>
      {options ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full text-sm px-3 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white text-slate-700"
        >
          <option value="">— Select —</option>
          {options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full text-sm px-3 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white placeholder:text-slate-300"
        />
      )}
    </div>
  );
}

export default function RecordDetailDrawer({ record, open, onClose, onFeedbackDone, onPatientSaved }: Props) {
  const [feedbackOpen, setFeedbackOpen]   = useState(false);
  const [feedbackDone, setFeedbackDone]   = useState<"correct" | "wrong" | null>(null);
  const [showResult, setShowResult]       = useState(true);
  const [editing, setEditing]             = useState(false);
  const [saving, setSaving]               = useState(false);
  const [saveError, setSaveError]         = useState<string | null>(null);

  // Patient form state
  const [name,   setName]   = useState("");
  const [patId,  setPatId]  = useState("");
  const [age,    setAge]    = useState("");
  const [gender, setGender] = useState("");
  const [notes,  setNotes]  = useState("");

  // Sync form fields when record changes or edit mode opens
  useEffect(() => {
    if (record) {
      setName(record.patient_name ?? "");
      setPatId(record.patient_id ?? "");
      setAge(record.patient_age != null ? String(record.patient_age) : "");
      setGender(record.patient_gender ?? "");
      setNotes(record.notes ?? "");
    }
    setEditing(false);
    setFeedbackDone(null);
    setSaveError(null);
  }, [record?.id]);

  const inferenceResult = record?.full_result as unknown as InferenceResponse | null;
  const hasFeedback = feedbackDone ?? (record?.feedback !== "pending" ? record?.feedback : null);

  async function handleCorrect() {
    if (!record) return;
    try {
      await recordsApi.submitFeedback(record.id, "correct");
    } catch { /* ignore */ }
    setFeedbackDone("correct");
    onFeedbackDone(record.id, "correct");
  }

  function handleModalDone(fb: "correct" | "wrong") {
    // FeedbackModal already called the API; just update local state
    setFeedbackOpen(false);
    setFeedbackDone(fb);
    if (record) onFeedbackDone(record.id, fb);
  }

  async function handleSave() {
    if (!record) return;
    setSaving(true);
    setSaveError(null);
    try {
      await recordsApi.updatePatient(record.id, {
        patient_name:   name,
        patient_id:     patId,
        patient_age:    age ? Number(age) : null,
        patient_gender: gender,
        notes,
      });
      setEditing(false);
      onPatientSaved?.(record.id, {
        patient_name:   name,
        patient_id:     patId,
        patient_age:    age ? Number(age) : null,
        patient_gender: gender,
        notes,
      });
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <AnimatePresence>
      {open && record && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Drawer */}
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className="fixed right-0 top-0 h-full z-50 w-full max-w-xl bg-slate-50 shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 bg-white border-b border-slate-100">
              <div>
                <h2 className="text-base font-semibold text-slate-800">
                  {record.patient_name || "Unnamed Patient"}
                </h2>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{record.request_id}</p>
              </div>
              <button
                onClick={onClose}
                className="text-slate-400 hover:text-slate-600 transition-colors p-1.5 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Scrollable body */}
            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">

              {/* Patient info card */}
              <div className="bg-white border border-slate-200 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Patient Info
                  </p>
                  {!editing && (
                    <button
                      onClick={() => setEditing(true)}
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 font-medium"
                    >
                      <Pencil className="w-3 h-3" /> Edit
                    </button>
                  )}
                </div>

                {editing ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <Field label="Full Name"  value={name}   onChange={setName}   placeholder="e.g. John Doe" />
                      <Field label="Patient ID" value={patId}  onChange={setPatId}  placeholder="e.g. P-00123" />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <Field label="Age (years)" value={age}    onChange={setAge}    type="number" placeholder="e.g. 45" />
                      <Field label="Gender"      value={gender} onChange={setGender} options={GENDER_OPTIONS} />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-500 mb-1">Clinical Notes</label>
                      <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        rows={3}
                        placeholder="Optional notes..."
                        className="w-full text-sm px-3 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white placeholder:text-slate-300 resize-none"
                      />
                    </div>
                    {saveError && (
                      <div className="flex items-center gap-2 text-xs text-red-600">
                        <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" /> {saveError}
                      </div>
                    )}
                    <div className="flex gap-2 pt-1">
                      <button
                        onClick={() => { setEditing(false); setSaveError(null); }}
                        className="flex-1 py-2 text-sm rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex-1 flex items-center justify-center gap-2 py-2 text-sm rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium transition-colors"
                      >
                        {saving
                          ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Saving…</>
                          : <><Check className="w-3.5 h-3.5" /> Save</>}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2 text-sm">
                    <Row label="Name"     value={name   || "—"} />
                    <Row label="ID"       value={patId  || "—"} />
                    <Row label="Age"      value={age    ? `${age} yrs` : "—"} />
                    <Row label="Gender"   value={gender || "—"} />
                    <Row label="Model"    value={record.model_version} />
                    <Row label="Analysed" value={new Date(record.created_at).toLocaleString("en-GB", {
                      year: "numeric", month: "short", day: "2-digit",
                      hour: "2-digit", minute: "2-digit",
                    })} />
                    {notes && (
                      <div className="mt-2 px-3 py-2 bg-slate-50 rounded-xl text-xs text-slate-600 border border-slate-100">
                        {notes}
                      </div>
                    )}
                    {!name && !patId && !age && !gender && (
                      <p className="text-xs text-slate-400 italic">No patient details yet — click Edit to add.</p>
                    )}
                  </div>
                )}
              </div>

              {/* Clinician feedback */}
              <div className="bg-white border border-slate-200 rounded-2xl p-4">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Clinician Feedback
                </p>
                {hasFeedback ? (
                  <div className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium ${
                    hasFeedback === "correct"
                      ? "bg-green-50 text-green-700 border border-green-200"
                      : "bg-amber-50 text-amber-700 border border-amber-200"
                  }`}>
                    {hasFeedback === "correct" ? "✓ Marked as correct" : `✗ Corrected → ${record.true_label ?? feedbackDone}`}
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={handleCorrect}
                      className="flex-1 py-2 text-sm font-medium rounded-xl border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 transition-colors"
                    >
                      ✓ Correct
                    </button>
                    <button
                      onClick={() => setFeedbackOpen(true)}
                      className="flex-1 py-2 text-sm font-medium rounded-xl border border-red-200 bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
                    >
                      ✗ Wrong
                    </button>
                  </div>
                )}
              </div>

              {/* Full inference result */}
              {inferenceResult ? (
                <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
                  <button
                    onClick={() => setShowResult((v) => !v)}
                    className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:bg-slate-50 transition-colors"
                  >
                    Full Analysis Result
                    {showResult
                      ? <ChevronUp className="w-4 h-4" />
                      : <ChevronDown className="w-4 h-4" />}
                  </button>
                  <AnimatePresence>
                    {showResult && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden px-4 pb-4"
                      >
                        <ResultsPanel
                          result={inferenceResult}
                          predictionId={record.id}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2 py-10 text-slate-400">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span className="text-sm">No saved analysis data</span>
                </div>
              )}
            </div>
          </motion.aside>

          <FeedbackModal
            predictionId={record.id}
            predictedLabel={record.primary_label}
            open={feedbackOpen}
            onClose={() => setFeedbackOpen(false)}
            onDone={handleModalDone}
          />
        </>
      )}
    </AnimatePresence>
  );
}
