import { ClipboardList } from "lucide-react";
import type { AuditRow } from "../types";

interface Props {
  rows: AuditRow[];
}

function DecisionBadge({ decision }: { decision: AuditRow["decision"] }) {
  const cls =
    decision === "positive"
      ? "bg-blue-50 text-blue-700 border-blue-200"
      : decision === "review_required" || decision === "review"
      ? "bg-amber-50 text-amber-700 border-amber-200"
      : decision === "likely_normal_or_uncertain"
      ? "bg-slate-50 text-slate-500 border-slate-200"
      : "bg-emerald-50 text-emerald-700 border-emerald-200";
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${cls}`}>
      {decision.charAt(0).toUpperCase() + decision.slice(1)}
    </span>
  );
}

export default function AuditTable({ rows }: Props) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl shadow-card overflow-hidden">
      <div className="flex items-center gap-2 px-5 py-3.5 border-b border-slate-100">
        <ClipboardList className="w-4 h-4 text-slate-400" />
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Recent Studies
        </p>
        <span className="ml-auto text-xs text-slate-400">
          {rows.length} records
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100">
              {["Timestamp", "Prediction", "Decision", "Confidence", "Request ID"].map(
                (h) => (
                  <th
                    key={h}
                    className="px-5 py-2.5 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider"
                  >
                    {h}
                  </th>
                )
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {rows.map((row) => (
              <tr key={row.id} className="hover:bg-slate-50 transition-colors">
                <td className="px-5 py-3 text-xs text-slate-500 font-mono whitespace-nowrap">
                  {row.timestamp}
                </td>
                <td className="px-5 py-3 text-xs font-medium text-slate-700">
                  {row.prediction.replace(/_/g, " ")}
                </td>
                <td className="px-5 py-3">
                  <DecisionBadge decision={row.decision} />
                </td>
                <td className="px-5 py-3">
                  <div className="flex items-center gap-2">
                    <div className="w-20 bg-slate-100 rounded-full h-1.5">
                      <div
                        className="bg-blue-400 h-1.5 rounded-full"
                        style={{ width: `${row.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-600 tabular-nums">
                      {(row.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-5 py-3 text-xs text-slate-400 font-mono">
                  {row.request_id}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
