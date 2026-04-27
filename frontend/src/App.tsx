import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import ImageViewer from "./components/ImageViewer";
import ResultsPanel, { ExplanationPanel } from "./components/ResultsPanel";
import ChatPanel from "./components/ChatPanel";
import HistoryPage from "./components/HistoryPage";
import LoginPage from "./components/LoginPage";
import RegisterPage from "./components/RegisterPage";
import { useAuth } from "./context/AuthContext";
import { predictApi, recordsApi } from "./services/api";
import type { AuditRow, InferenceResponse } from "./types";

type Page = "main" | "history";

function responseToAuditRow(r: InferenceResponse): AuditRow {
  return {
    id: r.request_id,
    timestamp: new Date().toLocaleString("en-GB", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    }),
    prediction: r.primary_prediction.label,
    decision: r.primary_prediction.decision,
    confidence: r.primary_prediction.calibrated_score,
    request_id: r.request_id,
  };
}

const SESSION_PRED_ID_KEY = "medxai_last_prediction_id";
const SESSION_IMAGE_KEY   = "medxai_last_image";
const MAX_IMAGE_BYTES     = 4 * 1024 * 1024; // 4 MB — skip caching larger files

function AuthenticatedApp() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(() => {
    // Restore image data-URL from previous session if available
    return sessionStorage.getItem(SESSION_IMAGE_KEY) ?? null;
  });
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [predictionId, setPredictionId] = useState<string | undefined>(() =>
    sessionStorage.getItem(SESSION_PRED_ID_KEY) ?? undefined
  );
  const [savedFeedback, setSavedFeedback] = useState<"correct" | "wrong" | undefined>(undefined);
  const [auditRows, setAuditRows] = useState<AuditRow[]>([]);
  const [showOverlay, setShowOverlay] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [oodEnabled, setOodEnabled] = useState(true);
  const [pathologyEnabled, setPathologyEnabled] = useState(true);
  const [localizationEnabled, setLocalizationEnabled] = useState(true);
  const [currentPage, setCurrentPage] = useState<Page>("main");

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    // Revoke old blob URL only if it was a blob (not a data URL restored from storage)
    if (imageUrl && imageUrl.startsWith("blob:")) URL.revokeObjectURL(imageUrl);
    const blobUrl = URL.createObjectURL(f);
    setImageUrl(blobUrl);
    setResult(null);
    setPredictionId(undefined);
    setSavedFeedback(undefined);
    setError(null);
    setShowOverlay(false);
    // Persist image as data-URL for page-refresh survival (skip if too large)
    if (f.size <= MAX_IMAGE_BYTES) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string | null;
        if (dataUrl) sessionStorage.setItem(SESSION_IMAGE_KEY, dataUrl);
      };
      reader.readAsDataURL(f);
    } else {
      sessionStorage.removeItem(SESSION_IMAGE_KEY);
    }
    sessionStorage.removeItem(SESSION_PRED_ID_KEY);
  }, [imageUrl]);

  const runEndpoint = useCallback(async (ep: "/v1/predict" | "/v1/explain") => {
    if (!file) return;
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const data: InferenceResponse & { db_id?: string } =
        ep === "/v1/predict"
          ? await predictApi.predict(file)
          : await predictApi.explain(file);
      setResult(data);
      setPredictionId(data.db_id ?? undefined);
      setSavedFeedback(undefined); // fresh prediction — no feedback yet
      setAuditRows((prev) => [responseToAuditRow(data), ...prev].slice(0, 50));
      // Only persist the DB id — result is fetched from DB on refresh
      if (data.db_id) sessionStorage.setItem(SESSION_PRED_ID_KEY, data.db_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsAnalyzing(false);
    }
  }, [file]);

  const handleAnalyze = useCallback(() => runEndpoint("/v1/predict"), [runEndpoint]);
  const handleExplain = useCallback(() => runEndpoint("/v1/explain"), [runEndpoint]);
 

  const handleReset = useCallback(() => {
    setFile(null);
    if (imageUrl && imageUrl.startsWith("blob:")) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setResult(null);
    setPredictionId(undefined);
    setSavedFeedback(undefined);
    setError(null);
    setShowOverlay(false);
    setIsAnalyzing(false);
    sessionStorage.removeItem(SESSION_PRED_ID_KEY);
    sessionStorage.removeItem(SESSION_IMAGE_KEY);
  }, [imageUrl]);

  // On mount: restore last result from DB if a predictionId was saved
  useEffect(() => {
    const savedId = sessionStorage.getItem(SESSION_PRED_ID_KEY);
    if (!savedId || result) return;   // nothing to restore, or already set
    recordsApi.get(savedId)
      .then((record) => {
        if (record.full_result) {
          setResult(record.full_result as unknown as InferenceResponse);
          setPredictionId(record.id);
          if (record.feedback === "correct" || record.feedback === "wrong") {
            setSavedFeedback(record.feedback);
          }
        }
      })
      .catch(() => {
        // record not found or auth expired — clear stale id
        sessionStorage.removeItem(SESSION_PRED_ID_KEY);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (currentPage === "history") {
    return (
      <div className="min-h-screen bg-slate-50">
        <Header currentPage={currentPage} onNavigate={setCurrentPage} historyCount={auditRows.length} />
        <HistoryPage rows={auditRows} onBack={() => setCurrentPage("main")} />
        <ChatPanel context={result?.primary_prediction ?? null} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Header currentPage={currentPage} onNavigate={setCurrentPage} historyCount={auditRows.length} />

      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        <div className="flex gap-5 items-start">

          {/* ── Left sidebar ── */}
          <aside className="w-72 flex-shrink-0">
            <UploadPanel
              file={file}
              isAnalyzing={isAnalyzing}
              oodEnabled={oodEnabled}
              pathologyEnabled={pathologyEnabled}
              localizationEnabled={localizationEnabled}
              onFileSelect={handleFileSelect}
              onAnalyze={handleAnalyze}
              onExplain={handleExplain}
              onReset={handleReset}
              onToggleOod={setOodEnabled}
              onTogglePathology={setPathologyEnabled}
              onToggleLocalization={setLocalizationEnabled}
            />
          </aside>

          {/* ── Center: image viewer ── */}
          <section className="flex-1 min-w-0">
            <ImageViewer
              imageUrl={imageUrl}
              result={result}
              showOverlay={showOverlay}
              onToggleOverlay={() => setShowOverlay((v) => !v)}
            />

            {/* Explanation panel — below image */}
            <AnimatePresence>
              {result && (
                <motion.div
                  key="explanation"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                  className="mt-4"
                >
                  <ExplanationPanel result={result} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Inference progress */}
            <AnimatePresence>
              {isAnalyzing && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="mt-4 bg-white border border-slate-200 rounded-2xl p-5 shadow-card"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
                    <p className="text-sm text-slate-600 font-medium">
                      Running inference — OOD check · Primary model · Calibration · Threshold…
                    </p>
                  </div>
                  <div className="mt-3 space-y-2">
                    {[80, 60, 40].map((w) => (
                      <div
                        key={w}
                        className="h-3 bg-slate-100 rounded-full animate-pulse"
                        style={{ width: `${w}%` }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error banner */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mt-4 flex items-start gap-3 bg-red-50 border border-red-200 rounded-2xl px-4 py-3"
                >
                  <span className="text-red-500 mt-0.5 text-base">⚠</span>
                  <div>
                    <p className="text-sm font-semibold text-red-700">Request failed</p>
                    <p className="text-xs text-red-600 mt-0.5 font-mono">{error}</p>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="ml-auto text-red-400 hover:text-red-600 text-lg leading-none"
                  >
                    ×
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          {/* ── Right sidebar: results ── */}
          <aside className="w-96 flex-shrink-0">
            <AnimatePresence>
              {result ? (
                <ResultsPanel result={result} predictionId={predictionId} initialFeedback={savedFeedback} />
              ) : !isAnalyzing && !error ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  className="bg-white border border-slate-200 rounded-2xl shadow-card p-8 flex flex-col items-center gap-3 text-center"
                >
                  <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center">
                    <svg className="w-6 h-6 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M9 17v-6h6v6m-3-9V3m0 0L9 6m3-3 3 3" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-500">No results yet</p>
                    <p className="text-xs text-slate-400 mt-1">Upload a study and click Analyse</p>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </aside>
        </div>

      </main>

      <ChatPanel context={result?.primary_prediction ?? null} />
    </div>
  );
}

export default function App() {
  const { user, loading: authLoading } = useAuth();
  const [authView, setAuthView] = useState<"login" | "register">("login");

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
          <p className="text-sm text-slate-400">Restoring session…</p>
        </div>
      </div>
    );
  }

  if (!user) {
    if (authView === "register") {
      return <RegisterPage onLoginClick={() => setAuthView("login")} />;
    }
    return <LoginPage onRegisterClick={() => setAuthView("register")} />;
  }

  return <AuthenticatedApp />;
}
