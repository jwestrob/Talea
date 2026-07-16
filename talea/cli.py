"""Command-line interface for Talea."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import transformers

from talea import __version__
from talea.artifacts import load_deployment, sha256_file
from talea.inference import TaleaPredictor
from talea.io import read_fasta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talea",
        description=(
            "Rank canonical protein sequences for elongated solenoid-like "
            "repeat architecture using frozen ESM++ attention models."
        ),
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="protein FASTA")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="prediction JSONL"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or an explicit torch device",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp16", "fp32", "bf16"),
        default="auto",
    )
    parser.add_argument("--maximum-batch", type=int, default=64)
    parser.add_argument("--square-budget", type=int, default=4_000_000)
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "allow the pinned ESM++ Hugging Face revision to load its model code"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the pinned ESM++ model to exist in the local HF cache",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deployment = load_deployment()
    records = read_fasta(
        args.input,
        maximum_length=int(deployment["input_contract"]["maximum_residues"]),
    )
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input FASTA and output JSONL must be different files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    audit_path = args.output.with_name(args.output.name + ".audit.json")
    existing = [path for path in (args.output, partial, audit_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite: " + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    started = time.time()
    predictor = TaleaPredictor(
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    completed = 0
    try:
        with partial.open("x") as handle:
            for row in predictor.iter_predictions(
                records,
                square_budget=args.square_budget,
                maximum_batch=args.maximum_batch,
                postprocess_workers=args.postprocess_workers,
                progress=not args.quiet,
            ):
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                completed += 1
        partial.replace(args.output)
    except Exception:
        print(
            f"Talea failed after {completed}/{len(records)} proteins; "
            f"partial output remains at {partial}",
            file=sys.stderr,
        )
        raise

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "talea_version": __version__,
        "schema_version": deployment["schema_version"],
        "primary_score_field": deployment["reporting_contract"]["rank_by"],
        "records": completed,
        "input_fasta": str(args.input),
        "output_jsonl": str(args.output),
        "input_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
        "model": deployment["esmplusplus"]["model"],
        "model_revision": deployment["esmplusplus"]["revision"],
        "device": str(predictor.device),
        "dtype": predictor.dtype_name,
        "model_loaded_seconds": predictor.loaded_seconds,
        "elapsed_seconds": time.time() - started,
        "canonical_amino_acid_alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "maximum_length": predictor.maximum_length,
        "runtime_artifact_sha256": predictor.assets.sha256,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
    }
    with audit_path.open("x") as handle:
        handle.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(
            f"Talea wrote {completed} predictions to {args.output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
