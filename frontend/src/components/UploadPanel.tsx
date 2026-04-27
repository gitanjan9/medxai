import { useCallback, useRef } from "react";
import { Upload, FileImage, RotateCcw, Scan, Layers, Eye, X } from "lucide-react";

interface Toggle {
  label: string;
  enabled: boolean;
  onToggle: (v: boolean) => void;
  icon: React.ReactNode;
}

interface UploadPanelProps {
  file: File | null;
  isAnalyzing: boolean;
  oodEnabled: boolean;
  pathologyEnabled: boolean;
  localizationEnabled: boolean;
  onFileSelect: (f: File) => void;
  onAnalyze: () => void;
  onExplain: () => void;
  onReset: () => void;
  onToggleOod: (v: boolean) => void;
  onTogglePathology: (v: boolean) => void;
  onToggleLocalization: (v: boolean) => void;
}

function ToggleChip({ label, enabled, onToggle, icon }: Toggle) {
  return (
    <button
      onClick={() => onToggle(!enabled)}
      className={`flex items-center gap-2 w-full px-3 py-2 rounded-xl border text-xs font-medium transition-all ${
        enabled
          ? "bg-blue-50 border-blue-200 text-blue-700"
          : "bg-white border-slate-200 text-slate-400"
      }`}
    >
      <span className={enabled ? "text-blue-500" : "text-slate-300"}>{icon}</span>
      <span className="flex-1 text-left">{label}</span>
      <span
        className={`w-7 h-4 rounded-full relative transition-all ${
          enabled ? "bg-blue-500" : "bg-slate-200"
        }`}
      >
        <span
          className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-all ${
            enabled ? "left-3.5" : "left-0.5"
          }`}
        />
      </span>
    </button>
  );
}

export default function UploadPanel({
  file,
  isAnalyzing,
  oodEnabled,
  pathologyEnabled,
  localizationEnabled,
  onFileSelect,
  onAnalyze,
  onExplain,
  onReset,
  onToggleOod,
  onTogglePathology,
  onToggleLocalization,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) onFileSelect(f);
    },
    [onFileSelect]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) onFileSelect(f);
    },
    [onFileSelect]
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Upload zone */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-card">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Study Upload
        </p>

        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
            file
              ? "border-blue-300 bg-blue-50"
              : "border-slate-200 hover:border-blue-300 hover:bg-slate-50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleChange}
          />
          {file ? (
            <div className="flex flex-col items-center gap-2">
              <FileImage className="w-8 h-8 text-blue-500" />
              <p className="text-xs font-medium text-slate-700 break-all">
                {file.name}
              </p>
              <p className="text-xs text-slate-400">
                {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-slate-300" />
              <p className="text-xs font-medium text-slate-500">
                Drop X-ray or click to browse
              </p>
              <p className="text-xs text-slate-400">JPEG · PNG · BMP · TIFF</p>
            </div>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-card flex flex-col gap-2">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
          Actions
        </p>

        <button
          onClick={onAnalyze}
          disabled={!file || isAnalyzing}
          className="flex items-center justify-center gap-2 w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-100 disabled:text-slate-300 text-white text-sm font-semibold rounded-xl transition-colors"
        >
          {isAnalyzing ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analysing…
            </>
          ) : (
            <>
              <Scan className="w-4 h-4" />
              Analyse Study
            </>
          )}
        </button>

        <button
          onClick={onExplain}
          disabled={!file || isAnalyzing}
          className="flex items-center justify-center gap-2 w-full py-2.5 bg-white hover:bg-slate-50 disabled:text-slate-300 text-blue-600 border border-blue-200 disabled:border-slate-200 text-sm font-semibold rounded-xl transition-colors"
        >
          <Eye className="w-4 h-4" />
          Explain + Localise
        </button>

        {file && (
          <button
            onClick={onReset}
            className="flex items-center justify-center gap-2 w-full py-2 text-slate-400 hover:text-slate-600 text-sm rounded-xl transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        )}
      </div>

      {/* Feature toggles */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-card flex flex-col gap-2">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
          Active Features
        </p>
        <ToggleChip
          label="OOD Detection"
          enabled={oodEnabled}
          onToggle={onToggleOod}
          icon={<X className="w-3 h-3" />}
        />
        <ToggleChip
          label="Pathology Model"
          enabled={pathologyEnabled}
          onToggle={onTogglePathology}
          icon={<Layers className="w-3 h-3" />}
        />
        <ToggleChip
          label="Localisation"
          enabled={localizationEnabled}
          onToggle={onToggleLocalization}
          icon={<Eye className="w-3 h-3" />}
        />
      </div>
    </div>
  );
}
