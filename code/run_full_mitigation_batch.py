from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_MODELS: List[str] = [
    "gemma2:9b",
    "qwen2.5-coder:7b-instruct",
    "deepseek-r1:8b",
    "llama3.1:8b",
    "mistral:v0.3",
]


def slugify_model(model: str) -> str:
    s = model.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_cmd(args: argparse.Namespace, model: str, out_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(args.worker_script),
        "--dataset", str(args.dataset),
        "--benchmark", str(args.benchmark),
        "--model", model,
        "--out", str(out_path),
        "--delay", str(args.delay),
    ]

    # framework flags
    if args.enable_filter:
        cmd.append("--enable_filter")
    if args.enable_aug:
        cmd.append("--enable_aug")
    if args.flip_case3:
        cmd.append("--flip_case3")
    if args.fallback_to_base:
        cmd.append("--fallback_to_base")

    # numeric args
    cmd += ["--n_aug", str(args.n_aug)]
    cmd += ["--max_rounds", str(args.max_rounds)]
    cmd += ["--timeout_base", str(args.timeout_base)]
    cmd += ["--timeout_call", str(args.timeout_call)]

    # cache
    if args.aug_cache:
        cmd += ["--aug_cache", str(args.aug_cache)]
    if args.baseline_jsonl:
        cmd += ["--baseline_jsonl", str(args.baseline_jsonl)]
    if args.skip_inference:
        cmd += ["--skip_inference"]
    if args.aug_model:
        cmd += ["--aug_model", args.aug_model]

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, type=Path, help="*_bugs.jsonl path (MBPP / HumanEval / QuixBugs etc.)")
    ap.add_argument("--benchmark", required=True, help="benchmark name (for bookkeeping in outputs)")
    ap.add_argument("--out_dir", required=True, type=Path, help="directory to store outputs (one JSON/JSONL per model)")

    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "models to run (space-separated). "
            "If omitted, runs the default 5 models: " + ", ".join(DEFAULT_MODELS)
        ),
    )

    # forwarding options to v3
    ap.add_argument("--worker_script", type=Path, default=Path(__file__).with_name("full_mitigation_framework.py"),
                    help="path to full_mitigation_framework.py")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--baseline_jsonl", type=Path, default=None, help="Optional baseline JSONL from main.py; when --skip_inference is set, reuse raw/pred instead of calling the judge model.")
    ap.add_argument("--skip_inference", action="store_true", help="Skip judge.check_requirement() when a valid cached result exists in --baseline_jsonl.")
    ap.add_argument("--aug_model", default="", help="model used to generate augmented tests; defaults to AUG_MODEL_NAME or deepseek-chat")

    ap.add_argument("--enable_filter", action="store_true")
    ap.add_argument("--enable_aug", action="store_true")
    ap.add_argument("--n_aug", type=int, default=12)
    ap.add_argument("--max_rounds", type=int, default=2)
    ap.add_argument("--timeout_base", type=float, default=3.0)
    ap.add_argument("--timeout_call", type=float, default=2.0)
    ap.add_argument("--aug_cache", type=Path, default=None, help="shared aug cache JSONL (recommended per-benchmark)")

    ap.add_argument("--flip_case3", action="store_true")
    ap.add_argument("--fallback_to_base", action="store_true")

    ap.add_argument("--continue_on_error", action="store_true",
                    help="if a model run crashes, continue to the next model instead of stopping")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.worker_script.exists():
        raise FileNotFoundError(f"worker_script not found: {args.worker_script}")

    models = args.models if args.models else DEFAULT_MODELS

    print("[INFO] Running models:")
    for m in models:
        print("  -", m)
    print()

    for m in models:
        slug = slugify_model(m)
        out_path = args.out_dir / f"{args.benchmark}__{slug}__full_filtered.json"

        cmd = build_cmd(args, m, out_path)
        print("=" * 100)
        print(f"[RUN] model={m}")
        print(f"[OUT]  {out_path}")
        print("[CMD] ", " ".join(cmd))
        print("=" * 100)

        try:
            # Let v3 handle resume via its own checkpoint JSONL.
            # We don't capture output so you can see progress live.
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] model={m} exited with code {e.returncode}")
            if not args.continue_on_error:
                raise
            print("[WARN] continue_on_error=True -> continue to next model")

    print("\n[DONE] All requested model runs finished (or were attempted).")
    print("If interrupted, re-run the same command; v3 will resume from its checkpoint JSONL files.")


if __name__ == "__main__":
    main()
