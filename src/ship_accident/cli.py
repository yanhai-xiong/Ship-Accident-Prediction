from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ship_accident.config import load_config
from ship_accident.train import run_training


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train ship accident classifiers (tabular + optional TF-IDF/BERT)."
    )
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    p.add_argument("--data", "-d", type=str, required=True)
    p.add_argument("--mode", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode

    result = run_training(cfg, data_path=args.data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
