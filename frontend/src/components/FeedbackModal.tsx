import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle2, XCircle, X, Loader2, AlertCircle, RefreshCw,
} from "lucide-react";
import { recordsApi } from "../services/api";

const LABELS = [
  "Normal", "Pneumonia", "Pleural Effusion", "Cardiomegaly",
  "Atelectasis", "Consolidation", "Pneumothorax", "Edema",
  "Emphysema", "Fibrosis", "No Finding",
];

interface Props {
  predictionId: string;
  predictedLabel: string;
  open: boolean;
  onClose: () => void;
  onDone: (feedback: "correct" | "wrong", retraining: boolean) => void;
}

export default function FeedbackModal({
  predictionId, predictedLabel, open, onClose, onDone,
}: Props) {
  const [step, setStep]         = useState<"choice" | "label" | "done">("choice");
  const [trueLabel, setTrueLabel] = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [retrained, setRetrained] = useState(false);

  async function handleCorrect() {
    setLoading(true);
    setError(null);
    try {
      await recordsApi.submitFeedback(predictionId, "correct");
      onDone("correct", false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setLoading(false);
    }
  }

  async function handleWrong() {
    if (!trueLabel) { setError("Please select the correct label"); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await recordsApi.submitFeedback(predictionId, "wrong", trueLabel);
      setRetrained(res.retraining_triggered);
      setStep("done");
      onDone("wrong", res.retraining_triggered);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setLoading(false);
    }
  }

  function handleClose() {
    setStep("choice");
    setTrueLabel("");
    setError(null);
    onClose();
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
          onClick={(e) => e.target === e.currentTarget && handleClose()}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2 }}
            className="bg-white rounded-2xl shadow-2xl w-full max-w-md"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
              <h2 className="text-base font-semibold text-slate-800">Clinician Feedback</h2>
              <button onClick={handleClose} className="text-slate-400 hover:text-slate-600 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Body */}
            <div className="px-6 py-5">
              {step === "choice" && (
                <>
                  <p className="text-sm text-slate-500 mb-1">Model predicted:</p>
                  <p className="text-base font-semibold text-slate-800 mb-5">{predictedLabel}</p>
                  <p className="text-sm text-slate-500 mb-3">Was this prediction correct?</p>
                  <div className="flex gap-3">
                    <button
                      onClick={handleCorrect}
                      disabled={loading}
                      className="flex-1 flex items-center justify-center gap-2 bg-green-50 hover:bg-green-100 border border-green-200 text-green-700 font-medium text-sm py-2.5 rounded-xl transition-colors"
                    >
                      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                      Correct
                    </button>
                    <button
                      onClick={() => setStep("label")}
                      disabled={loading}
                      className="flex-1 flex items-center justify-center gap-2 bg-red-50 hover:bg-red-100 border border-red-200 text-red-700 font-medium text-sm py-2.5 rounded-xl transition-colors"
                    >
                      <XCircle className="w-4 h-4" />
                      Wrong
                    </button>
                  </div>
                </>
              )}

              {step === "label" && (
                <>
                  <p className="text-sm text-slate-500 mb-3">Select the correct diagnosis:</p>
                  <div className="grid grid-cols-2 gap-2 mb-4 max-h-52 overflow-y-auto pr-1">
                    {LABELS.map((lbl) => (
                      <button
                        key={lbl}
                        onClick={() => setTrueLabel(lbl)}
                        className={`text-left text-sm px-3 py-2 rounded-xl border transition-colors ${
                          trueLabel === lbl
                            ? "bg-blue-600 border-blue-600 text-white font-medium"
                            : "bg-slate-50 border-slate-200 text-slate-700 hover:border-blue-300"
                        }`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>
                  {error && (
                    <div className="flex items-center gap-2 text-sm text-red-600 mb-3">
                      <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      onClick={() => setStep("choice")}
                      className="flex-1 text-sm py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors"
                    >
                      Back
                    </button>
                    <button
                      onClick={handleWrong}
                      disabled={loading || !trueLabel}
                      className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium text-sm py-2.5 rounded-xl transition-colors"
                    >
                      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                      Submit correction
                    </button>
                  </div>
                </>
              )}

              {step === "done" && (
                <div className="flex flex-col items-center gap-3 py-2 text-center">
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center ${retrained ? "bg-blue-100" : "bg-green-100"}`}>
                    {retrained
                      ? <RefreshCw className="w-6 h-6 text-blue-600" />
                      : <CheckCircle2 className="w-6 h-6 text-green-600" />}
                  </div>
                  <p className="font-semibold text-slate-800">Feedback saved</p>
                  {retrained ? (
                    <p className="text-sm text-blue-600">
                      Enough corrections accumulated — <strong>retraining queued</strong>.<br />
                      The model will improve automatically.
                    </p>
                  ) : (
                    <p className="text-sm text-slate-500">
                      Correction recorded as <strong>{trueLabel}</strong>.<br />
                      Retraining will trigger when more samples accumulate.
                    </p>
                  )}
                  <button
                    onClick={handleClose}
                    className="mt-1 text-sm text-slate-500 hover:text-slate-700 underline-offset-2 hover:underline"
                  >
                    Close
                  </button>
                </div>
              )}

              {step !== "done" && error && step === "choice" && (
                <p className="mt-3 text-xs text-red-500 flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5" /> {error}
                </p>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
