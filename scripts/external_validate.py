"""External dataset validation.

Evaluates a trained checkpoint on a held-out CSV not used during training.
Compares internal vs external AUROC/AUPRC/ECE and flags calibration drift.

Usage
-----
  python scripts/external_validate.py \\
      --checkpoint  artifacts/v3/checkpoints \\
      --internal    reports/v3/summary.json \\
      --external-data   /path/to/external_test.csv \\
      --label-map   artifacts/v3/label_map.json \\
      --out         reports/external_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.logging import get_logger
from src.evaluation.evaluate import run_evaluation

logger = get_logger("scripts.external_validate")

_AUROC_DROP_WARN = 0.05   # warn if external AUROC drops more than this
_ECE_DRIFT_WARN = 0.03    # warn if ECE increases more than this


def compare(internal: dict, external: dict) -> list[str]:
    warnings_out = []

    auroc_drop = internal.get("macro_auroc", 0) - external.get("macro_auroc", 0)
    if auroc_drop > _AUROC_DROP_WARN:
        warnings_out.append(
            f"⚠  AUROC drop: internal={internal['macro_auroc']:.4f}  "
            f"external={external['macro_auroc']:.4f}  Δ={auroc_drop:.4f}"
        )

    ece_drift = external.get("macro_ece", 0) - internal.get("macro_ece", 0)
    if ece_drift > _ECE_DRIFT_WARN:
        warnings_out.append(
            f"⚠  Calibration drift: ECE internal={internal['macro_ece']:.4f}  "
            f"external={external['macro_ece']:.4f}  Δ={ece_drift:.4f}"
        )

    return warnings_out


def main() -> None:
    p = argparse.ArgumentParser(description="External dataset validation.")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--external-data", required=True)
    p.add_argument("--label-map",     required=True)
    p.add_argument("--out",           default="reports/external")
    p.add_argument("--internal",      help="reports/v3/summary.json (optional, for comparison)")
    p.add_argument("--image-size",    type=int, default=320)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--device",        default="auto")
    args = p.parse_args()

    logger.info("Running external evaluation on: %s", args.external_data)
    ext_summary = run_evaluation(
        checkpoint_dir=args.checkpoint,
        data_csv=args.external_data,
        label_map_json=args.label_map,
        out_dir=args.out,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device_str=args.device,
    )

    report = {"external": ext_summary, "comparison_warnings": []}

    if args.internal and Path(args.internal).exists():
        with open(args.internal) as f:
            int_summary = json.load(f)
        report["internal"] = int_summary
        warns = compare(int_summary, ext_summary)
        report["comparison_warnings"] = warns
        print("\n── Comparison ──────────────────────────────────────────────")
        print(f"  Internal  AUROC={int_summary.get('macro_auroc'):.4f}  "
              f"AUPRC={int_summary.get('macro_auprc'):.4f}  "
              f"ECE={int_summary.get('macro_ece'):.4f}")
        print(f"  External  AUROC={ext_summary['macro_auroc']:.4f}  "
              f"AUPRC={ext_summary['macro_auprc']:.4f}  "
              f"ECE={ext_summary['macro_ece']:.4f}")
        if warns:
            print("\n  Warnings:")
            for w in warns:
                print(f"    {w}")
        else:
            print("  ✓ No significant drift detected.")

    out_path = Path(args.out) / "external_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", out_path)


if __name__ == "__main__":
    main()
