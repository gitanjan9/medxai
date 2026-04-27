import { useState, useEffect } from "react";
import { ZoomIn, ZoomOut, Maximize2, Minimize2, ToggleLeft, ToggleRight, AlertTriangle, X } from "lucide-react";
import type { InferenceResponse } from "../types";

interface ImageViewerProps {
  imageUrl: string | null;
  result: InferenceResponse | null;
  showOverlay: boolean;
  onToggleOverlay: () => void;
}

function ImageContent({
  imageUrl,
  zoom,
  showOverlay,
  hasLocalization,
  bbox,
  result,
  fullscreen,
}: {
  imageUrl: string;
  zoom: number;
  showOverlay: boolean;
  hasLocalization: boolean;
  bbox: InferenceResponse["localization"]["bbox"] | null | undefined;
  result: InferenceResponse | null;
  fullscreen: boolean;
}) {
  const BASE_W = fullscreen ? 720 : 400;
  const px = Math.round(BASE_W * zoom);

  return (
    <div className="relative inline-block flex-shrink-0" style={{ width: px }}>
      <img
        src={imageUrl}
        alt="Uploaded X-ray"
        className="w-full h-auto object-contain rounded"
        style={{ display: "block" }}
      />
      {showOverlay && hasLocalization && bbox && (
        <div
          className="absolute border-2 border-blue-400 bg-blue-400/10 rounded pointer-events-none"
          style={{
            left: `${(bbox.x1_norm ?? 0) * 100}%`,
            top: `${(bbox.y1_norm ?? 0) * 100}%`,
            width: `${((bbox.x2_norm ?? 1) - (bbox.x1_norm ?? 0)) * 100}%`,
            height: `${((bbox.y2_norm ?? 1) - (bbox.y1_norm ?? 0)) * 100}%`,
          }}
        >
          <span className="absolute -top-5 left-0 bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded whitespace-nowrap">
            {result?.localization?.region ?? "Attention region"}
          </span>
        </div>
      )}
    </div>
  );
}

export default function ImageViewer({
  imageUrl,
  result,
  showOverlay,
  onToggleOverlay,
}: ImageViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);

  const bbox = result?.localization?.bbox;
  const hasLocalization =
    result?.localization?.enabled &&
    result.localization.type === "approximate_attention_region" &&
    bbox != null;

  // Close fullscreen on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setFullscreen(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const ToolbarButtons = ({ inFullscreen }: { inFullscreen: boolean }) => (
    <div className="flex items-center gap-2">
      {hasLocalization && (
        <button
          onClick={onToggleOverlay}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors ${
            showOverlay ? "bg-blue-50 border-blue-200 text-blue-700" : "bg-white border-slate-200 text-slate-500"
          }`}
        >
          {showOverlay ? <ToggleRight className="w-3.5 h-3.5" /> : <ToggleLeft className="w-3.5 h-3.5" />}
          Attention Overlay
        </button>
      )}
      <button
        onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
        title="Zoom in"
        className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
      >
        <ZoomIn className="w-4 h-4" />
      </button>
      <button
        onClick={() => setZoom((z) => Math.max(z - 0.25, 0.25))}
        title="Zoom out"
        className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
      >
        <ZoomOut className="w-4 h-4" />
      </button>
      <button
        onClick={() => { setZoom(1); if (inFullscreen) setFullscreen(false); else setFullscreen(true); }}
        title={inFullscreen ? "Exit fullscreen" : "Fullscreen"}
        className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
      >
        {inFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
      </button>
      {zoom !== 1 && (
        <span className="text-xs text-slate-400 tabular-nums">{Math.round(zoom * 100)}%</span>
      )}
    </div>
  );

  return (
    <>
      {/* ── Inline viewer ── */}
      <div className="bg-white border border-slate-200 rounded-2xl shadow-card overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <div>
            <p className="text-xs font-semibold text-slate-700">Image Viewer</p>
            {imageUrl && <p className="text-xs text-slate-400 mt-0.5">Study loaded</p>}
          </div>
          <ToolbarButtons inFullscreen={false} />
        </div>

        <div
          className="relative bg-slate-100 overflow-auto"
          style={{ minHeight: 420 }}
        >
          {imageUrl ? (
            <div style={{ minWidth: "100%", minHeight: 420, display: "flex", alignItems: "center", justifyContent: "center", padding: "16px" }}>
              <ImageContent
                imageUrl={imageUrl}
                zoom={zoom}
                showOverlay={showOverlay}
                hasLocalization={!!hasLocalization}
                bbox={bbox}
                result={result}
                fullscreen={false}
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-3 text-slate-300" style={{ minHeight: 420 }}>
              <svg className="w-16 h-16" fill="none" viewBox="0 0 64 64" stroke="currentColor" strokeWidth={1}>
                <rect x="8" y="8" width="48" height="48" rx="4" />
                <circle cx="32" cy="26" r="8" />
                <path d="M12 56 c0-10 8-18 20-18s20 8 20 18" />
              </svg>
              <p className="text-sm font-medium text-slate-400">Upload a chest X-ray to begin</p>
            </div>
          )}
        </div>

        {hasLocalization && (
          <div className="flex items-start gap-2 px-4 py-2.5 bg-amber-50 border-t border-amber-100">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-700">{result?.localization?.disclaimer}</p>
          </div>
        )}
      </div>

      {/* ── Fullscreen lightbox ── */}
      {fullscreen && imageUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex flex-col"
          onClick={() => setFullscreen(false)}
        >
          {/* Fullscreen toolbar */}
          <div
            className="flex items-center justify-between px-6 py-4 bg-black/40 flex-shrink-0"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="text-sm font-semibold text-white">Image Viewer — Fullscreen</p>
            <div className="flex items-center gap-2">
              <ToolbarButtons inFullscreen={true} />
              <button
                onClick={() => setFullscreen(false)}
                className="ml-2 p-1.5 rounded-lg hover:bg-white/10 text-white transition-colors"
                title="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Fullscreen image */}
          <div
            className="flex-1 overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ minWidth: "100%", minHeight: "100%", display: "flex", alignItems: "center", justifyContent: "center", padding: "24px" }}>
              <ImageContent
                imageUrl={imageUrl}
                zoom={zoom}
                showOverlay={showOverlay}
                hasLocalization={!!hasLocalization}
                bbox={bbox}
                result={result}
                fullscreen={true}
              />
            </div>
          </div>

          <p className="text-center text-xs text-white/40 pb-3">
            Press Esc or click outside to close
          </p>
        </div>
      )}
    </>
  );
}
