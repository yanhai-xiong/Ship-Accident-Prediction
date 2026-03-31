#!/usr/bin/env python3
"""Run all training configs on one spreadsheet and write JSON + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ship_accident.config import load_config
from ship_accident.train import run_training

RUNS: list[tuple[str, str]] = [
    ("configs/legacy_excel_model1a.yaml", "model1a_tabular.json"),
    ("configs/legacy_excel_model1a_tfidf.yaml", "model1a_tfidf.json"),
    ("configs/legacy_excel_model3a.yaml", "model3a_accident_level.json"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", type=Path, default=ROOT / "all_data_0710.xlsx")
    p.add_argument("--out-dir", "-o", type=Path, default=ROOT / "reports")
    args = p.parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        print(f"ERROR: not found: {data_path}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    for cfg_rel, json_name in RUNS:
        cfg_path = ROOT / cfg_rel
        cfg = load_config(cfg_path)
        print(f"Running {cfg_rel} ...", file=sys.stderr)
        result = run_training(cfg, data_path=data_path)
        out_json = out_dir / json_name
        out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        target = cfg.get("target_column", "?")
        mode = cfg.get("mode", "?")
        lines = [
            f"## {json_name}",
            f"- **Config**: `{cfg_rel}`",
            f"- **Target**: `{target}`",
            f"- **Mode**: `{mode}`",
            f"- **Train / test rows**: {result['n_train']} / {result['n_test']}",
            f"- **Feature dim**: {result['n_features']}",
            "",
            "| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |",
            "|-------|---------------|----------|---------------|---------------|",
        ]
        for mname, m in result["models"].items():
            lines.append(
                f"| {mname} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                f"{m['weighted_f1']:.4f} | {m['best_cv_score']:.4f} |"
            )
        lines.append("")
        sections.append("\n".join(lines))

    report = (
        "# Training report\n\n"
        f"Data file: `{data_path}`\n\n"
        + "\n".join(sections)
        + "\n## Classification reports (test set)\n\n"
    )
    for _, json_name in RUNS:
        payload = json.loads((out_dir / json_name).read_text(encoding="utf-8"))
        report += f"### {json_name}\n\n"
        for mname, m in payload["models"].items():
            report += f"#### {mname}\n\n```\n{m['classification_report']}\n```\n\n"

    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir}/REPORT.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
