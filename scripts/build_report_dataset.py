#!/usr/bin/env python3
"""Scan a folder of Mandarin accident reports, extract text, infer labels, write CSV for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from ship_accident.report_io import extract_report_text
from ship_accident.report_parse import (
    engineer_post_accident_features,
    engineer_report_features,
    infer_accident_type,
    resolve_accident_severity_training_label,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Folder with .pdf / .doc / .docx / .wps reports",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "data" / "report_corpus.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--pdf-max-pages",
        type=int,
        default=None,
        help="Limit PDF pages read (default: all). Use e.g. 25 for faster iteration.",
    )
    p.add_argument("--max-files", type=int, default=None, help="Process only first N files (debug).")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON path listing per-file status and parse notes",
    )
    args = p.parse_args()

    reports_dir = args.reports_dir.expanduser().resolve()
    if not reports_dir.is_dir():
        print(f"ERROR: not a directory: {reports_dir}", file=sys.stderr)
        return 1

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    exts = {".pdf", ".doc", ".docx", ".wps"}
    files = sorted(
        f for f in reports_dir.iterdir() if f.is_file() and f.suffix.lower() in exts and not f.name.startswith(".")
    )
    if args.max_files:
        files = files[: args.max_files]

    rows: list[dict] = []
    manifest: list[dict] = []

    for i, path in enumerate(files):
        if i and i % 100 == 0:
            print(f"... {i}/{len(files)}", file=sys.stderr)
        rec: dict = {"file_id": path.stem, "path": str(path)}
        try:
            text = extract_report_text(path, pdf_max_pages=args.pdf_max_pages)
        except Exception as e:  # noqa: BLE001
            manifest.append({"file": path.name, "ok": False, "error": str(e)})
            continue
        if not text.strip():
            manifest.append({"file": path.name, "ok": False, "error": "empty_text"})
            continue
        acc_type, raw_title = infer_accident_type(text)
        feats = engineer_report_features(text)
        post = engineer_post_accident_features(text)
        sev, sev_src, sev_phrase = resolve_accident_severity_training_label(text, post)
        rec["accident type"] = acc_type if acc_type else ""
        rec["title_type_raw"] = raw_title or ""
        rec["accident severity"] = sev if sev else None
        rec["severity_label_source"] = sev_src if sev_src else None
        rec["severity_match_raw"] = sev_phrase if sev_phrase else None
        rec["full_text"] = text[:200_000]
        rec.update(feats)
        rec.update(post)
        rows.append(rec)
        manifest.append(
            {
                "file": path.name,
                "ok": True,
                "accident_type": acc_type,
                "accident_severity": sev,
                "severity_label_source": sev_src,
                "chars": len(text),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("ERROR: no rows built", file=sys.stderr)
        return 1
    before = len(df)
    df = df[df["accident type"].astype(str).str.len() > 0]
    dropped = before - len(df)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(
        f"Wrote {out} ({len(df)} rows, dropped {dropped} without label). "
        f"Columns: {list(df.columns)[:8]}...",
        file=sys.stderr,
    )
    if args.manifest:
        mp = args.manifest.expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote manifest {mp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
