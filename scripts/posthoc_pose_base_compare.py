#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_RUN_DIR = REPO_ROOT / "logs/pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544"
DEFAULT_BASE_LOG = REPO_ROOT / "logs/pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544.log"
DEFAULT_REPORT_MD = REPO_ROOT / "report.md"
DEFAULT_ANALYSIS_TAG = "internvl3"
DEFAULT_ATTRIBUTE_FILE = REPO_ROOT / "save/attribute_coco_pose_gemini_internvl3_8B.json"
DEFAULT_TRAIN_ATTRIBUTE_FILE = REPO_ROOT / "save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
DEFAULT_CORPUS_FILE = REPO_ROOT / "save/corpus_coco_pose_gemini.json"
DEFAULT_BASE_REFERENCE_DIR = REPO_ROOT / "logs/pose/base_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate posthoc failure-case and base-vs-run comparison reports for "
            "COCO pose resume runs."
        )
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=DEFAULT_BASE_RUN_DIR,
        help="Base pose run directory with val/test predictions and/or slice stats.",
    )
    parser.add_argument(
        "--base-log-file",
        type=Path,
        default=DEFAULT_BASE_LOG,
        help="Base training log to copy into a canonical base-reference folder.",
    )
    parser.add_argument(
        "--copy-base-log-dir",
        type=Path,
        default=DEFAULT_BASE_REFERENCE_DIR,
        help="Directory that will receive the copied base log.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=DEFAULT_REPORT_MD,
        help="Markdown report used to discover canonical ratio-run logs.",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit pose resume log(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Explicit pose resume artifact directory/directories.",
    )
    parser.add_argument(
        "--analysis-tag",
        type=str,
        default=DEFAULT_ANALYSIS_TAG,
        help="Slice-analysis subdirectory tag. Default: internvl3",
    )
    parser.add_argument(
        "--attribute-file",
        type=Path,
        default=DEFAULT_ATTRIBUTE_FILE,
        help="Eval-side attribute JSON/CSV for slice analysis.",
    )
    parser.add_argument(
        "--attribute-split",
        type=str,
        default="val_full",
        choices=["train", "val_full", "val", "test", "all"],
        help="Attribute split corresponding to --attribute-file.",
    )
    parser.add_argument(
        "--train-attribute-file",
        type=Path,
        default=DEFAULT_TRAIN_ATTRIBUTE_FILE,
        help="Train-side attribute JSON/CSV for slice analysis.",
    )
    parser.add_argument(
        "--train-attribute-split",
        type=str,
        default="train",
        choices=["train", "val_full", "val", "test", "all"],
        help="Attribute split corresponding to --train-attribute-file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help="Optional corpus JSON used to preserve attribute/value ordering.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/oxx220000/coco",
        help="COCO root for posthoc slice analysis.",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=20000,
        help="Train-image cap used by the original base run.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=25,
        help="Minimum images per slice during slice analysis.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=14,
        help="How many base failure cases to carry into the focused compare report.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["val", "test"],
        default=[],
        help="Split(s) to process. Default: test only.",
    )
    parser.add_argument(
        "--acc-diff-threshold",
        type=float,
        default=0.005,
        help="Slice failure AP-drop threshold versus overall AP.",
    )
    parser.add_argument(
        "--distribution-diff-threshold",
        type=float,
        default=0.15,
        help="Distribution-shift threshold used by slice analysis.",
    )
    parser.add_argument(
        "--rare-label-threshold",
        type=float,
        default=1.1,
        help="Rare-label multiplier used by slice analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for val/test half split reconstruction.",
    )
    parser.add_argument(
        "--force-analyze",
        action="store_true",
        help="Regenerate slice-analysis JSON/CSV even if they already exist.",
    )
    return parser.parse_args()


def ensure_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_report_logs(report_md: Path) -> List[Path]:
    text = report_md.read_text(encoding="utf-8")
    rel_paths = re.findall(r"`(logs/pose/pose_resume_internvl3_epoch2_[^`]+\.log)`", text)
    unique_paths: List[Path] = []
    seen = set()
    for rel_path in rel_paths:
        full_path = (REPO_ROOT / rel_path).resolve()
        if full_path in seen:
            continue
        seen.add(full_path)
        unique_paths.append(full_path)
    return unique_paths


def extract_artifact_dir_from_log(log_path: Path) -> Path:
    artifact_dir = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Training complete. Artifacts:" in line:
                artifact_dir = line.split("Training complete. Artifacts:", 1)[1].strip()
    if artifact_dir is None:
        raise ValueError(f"Could not find artifact directory in log: {log_path}")
    artifact_path = Path(artifact_dir)
    if not artifact_path.is_absolute():
        artifact_path = (REPO_ROOT / artifact_path).resolve()
    return artifact_path


def discover_run_dirs(args: argparse.Namespace) -> List[Path]:
    run_dirs: List[Path] = []
    seen = set()

    explicit_logs = [path.resolve() for path in args.run_log]
    explicit_dirs = [path.resolve() for path in args.run_dir]
    if explicit_logs or explicit_dirs:
        all_logs = explicit_logs
    else:
        all_logs = parse_report_logs(args.report_md)

    for log_path in all_logs:
        if not log_path.exists():
            raise FileNotFoundError(f"Missing run log: {log_path}")
        run_dir = extract_artifact_dir_from_log(log_path)
        if run_dir in seen:
            continue
        seen.add(run_dir)
        run_dirs.append(run_dir)

    for resolved in explicit_dirs:
        if resolved in seen:
            continue
        seen.add(resolved)
        run_dirs.append(resolved)

    if not run_dirs:
        raise ValueError("No run directories were discovered from report.md or explicit arguments.")

    return run_dirs


def copy_base_log(base_log: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = base_log.name
    if not filename.startswith("base_"):
        filename = f"base_{filename}"
    dest_path = dest_dir / filename
    shutil.copy2(base_log, dest_path)
    return dest_path


def analysis_dir(run_dir: Path, tag: str) -> Path:
    return run_dir / "slice_analysis" / tag


def split_paths(run_dir: Path, tag: str, split: str) -> Dict[str, Path]:
    out_dir = analysis_dir(run_dir, tag)
    return {
        "json": out_dir / f"{split}_slice_stats.json",
        "csv": out_dir / f"{split}_slice_stats.csv",
        "analysis_log": out_dir / f"{split}_posthoc_slice_analysis.log",
        "failure_md": out_dir / f"{split}_failure_cases_top16.md",
        "compare_md": out_dir / f"{split}_compare_vs_base_failure_set.md",
        "compare_csv": out_dir / f"{split}_compare_vs_base_failure_set.csv",
    }


def prediction_path(run_dir: Path, split: str) -> Path:
    if split == "val":
        return run_dir / "val_predictions_best.json"
    if split == "test":
        return run_dir / "test_predictions.json"
    raise ValueError(f"Unsupported split: {split}")


def run_slice_analysis(
    *,
    run_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> None:
    paths = split_paths(run_dir, args.analysis_tag, split)
    pred_path = prediction_path(run_dir, split)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file for {split}: {pred_path}")

    paths["analysis_log"].parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "utils/stat_pose_estimation.py"),
        "--predictions-json",
        str(pred_path),
        "--attribute-file",
        str(args.attribute_file),
        "--attribute-split",
        args.attribute_split,
        "--train-attribute-file",
        str(args.train_attribute_file),
        "--train-attribute-split",
        args.train_attribute_split,
        "--corpus",
        str(args.corpus),
        "--data-root",
        args.data_root,
        "--eval-split",
        split,
        "--train-max-samples",
        str(args.train_max_samples),
        "--min-images",
        str(args.min_images),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--acc-diff-threshold",
        str(args.acc_diff_threshold),
        "--distribution-diff-threshold",
        str(args.distribution_diff_threshold),
        "--rare-label-threshold",
        str(args.rare_label_threshold),
        "--output-json",
        str(paths["json"]),
        "--output-csv",
        str(paths["csv"]),
    ]

    with paths["analysis_log"].open("w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def ensure_slice_stats(
    *,
    run_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> Path:
    paths = split_paths(run_dir, args.analysis_tag, split)
    if args.force_analyze or not paths["json"].exists() or not paths["csv"].exists():
        run_slice_analysis(run_dir=run_dir, split=split, args=args)
    return paths["json"]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def failure_rows(summary: Dict) -> List[Dict]:
    return [row for row in summary.get("slices", []) if row.get("is_failure_case")]


def write_failure_report(summary: Dict, output_path: Path, *, top_k: int, run_dir: Path, split: str) -> None:
    overall = summary.get("overall") or {}
    rows = failure_rows(summary)[:top_k]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Failure Cases Report",
        "",
        f"- Run: `{ensure_repo_relative(run_dir)}`",
        f"- Split: `{split}`",
        f"- Predictions: `{summary.get('predictions_json', 'NA')}`",
        f"- Overall AP: `{format_float(overall.get('AP'))}`",
        f"- Overall AP50: `{format_float(overall.get('AP50'))}`",
        f"- Overall AR: `{format_float(overall.get('AR'))}`",
        f"- Total failure cases in summary: `{len(failure_rows(summary))}`",
        f"- Reported top-k: `{min(top_k, len(failure_rows(summary)))}`",
        "",
        "## Top Failure Cases",
        "",
    ]

    if not rows:
        lines.append("No failure cases met the configured threshold.")
    else:
        for idx, row in enumerate(rows, start=1):
            pieces = [
                f"{idx}. `{row['attribute']}={row['value']}`",
                f"images={format_int(row.get('num_images'))}",
                f"gt={format_int(row.get('num_gt_annotations'))}",
                f"pred={format_int(row.get('num_predictions'))}",
                f"AP={format_float(row.get('AP'))}",
                f"delta={format_float(row.get('ap_delta'), digits=4)}",
            ]
            if row.get("failure_type"):
                pieces.append(f"type={row['failure_type']}")
            if row.get("train_share") is not None and row.get("eval_share") is not None:
                pieces.extend(
                    [
                        f"train_share={format_float(row.get('train_share'))}",
                        f"eval_share={format_float(row.get('eval_share'))}",
                        f"gap={format_float(row.get('share_gap'), digits=4)}",
                    ]
                )
            lines.append(" - ".join(pieces))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slice_map(summary: Dict) -> Dict[Tuple[str, str], Dict]:
    return {(row["attribute"], row["value"]): row for row in summary.get("slices", [])}


def compare_status(run_ap: Optional[float], base_ap: Optional[float], eps: float = 1e-12) -> str:
    if run_ap is None or base_ap is None:
        return "missing"
    diff = run_ap - base_ap
    if diff > eps:
        return "improved"
    if diff < -eps:
        return "declined"
    return "same"


def compare_rows(base_summary: Dict, run_summary: Dict) -> List[Dict]:
    base_rows = slice_map(base_summary)
    run_rows = slice_map(run_summary)
    keys = sorted(set(base_rows) | set(run_rows))
    rows: List[Dict] = []
    for attribute, value in keys:
        base_row = base_rows.get((attribute, value), {})
        run_row = run_rows.get((attribute, value), {})
        base_ap = base_row.get("AP")
        run_ap = run_row.get("AP")
        ap_diff = None if base_ap is None or run_ap is None else round(run_ap - base_ap, 6)

        base_failure = bool(base_row.get("is_failure_case"))
        run_failure = bool(run_row.get("is_failure_case"))
        if base_failure and run_failure:
            failure_transition = "still_failure"
        elif base_failure and not run_failure:
            failure_transition = "resolved_failure"
        elif (not base_failure) and run_failure:
            failure_transition = "new_failure"
        else:
            failure_transition = "never_failure"

        rows.append(
            {
                "attribute": attribute,
                "value": value,
                "base_AP": base_ap,
                "run_AP": run_ap,
                "AP_delta_vs_base": ap_diff,
                "status_vs_base": compare_status(run_ap, base_ap),
                "base_ap_delta": base_row.get("ap_delta"),
                "run_ap_delta": run_row.get("ap_delta"),
                "base_is_failure_case": base_failure,
                "run_is_failure_case": run_failure,
                "failure_transition": failure_transition,
                "base_failure_type": base_row.get("failure_type"),
                "run_failure_type": run_row.get("failure_type"),
                "base_num_images": base_row.get("num_images"),
                "run_num_images": run_row.get("num_images"),
                "base_eval_share": base_row.get("eval_share"),
                "run_eval_share": run_row.get("eval_share"),
                "base_train_share": base_row.get("train_share"),
                "run_train_share": run_row.get("train_share"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["AP_delta_vs_base"] is None,
            row["AP_delta_vs_base"] if row["AP_delta_vs_base"] is not None else 0.0,
            row["attribute"],
            row["value"],
        )
    )
    return rows


def write_compare_csv(rows: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute",
        "value",
        "base_AP",
        "run_AP",
        "AP_delta_vs_base",
        "status_vs_base",
        "base_ap_delta",
        "run_ap_delta",
        "base_is_failure_case",
        "run_is_failure_case",
        "failure_transition",
        "base_failure_type",
        "run_failure_type",
        "base_num_images",
        "run_num_images",
        "base_eval_share",
        "run_eval_share",
        "base_train_share",
        "run_train_share",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if row.get(key) is None else row.get(key))
                    for key in fieldnames
                }
            )


def top_rows(rows: Sequence[Dict], *, kind: str, top_k: int) -> List[Dict]:
    filtered = [row for row in rows if row.get("AP_delta_vs_base") is not None]
    if kind == "improved":
        filtered = [row for row in filtered if row["AP_delta_vs_base"] > 0]
        filtered.sort(key=lambda row: row["AP_delta_vs_base"], reverse=True)
    elif kind == "declined":
        filtered = [row for row in filtered if row["AP_delta_vs_base"] < 0]
        filtered.sort(key=lambda row: row["AP_delta_vs_base"])
    elif kind == "resolved":
        filtered = [row for row in filtered if row["failure_transition"] == "resolved_failure"]
        filtered.sort(key=lambda row: row["AP_delta_vs_base"], reverse=True)
    elif kind == "new_failure":
        filtered = [row for row in filtered if row["failure_transition"] == "new_failure"]
        filtered.sort(key=lambda row: row["AP_delta_vs_base"])
    elif kind == "rare_to_not_rare":
        filtered = [
            row
            for row in filtered
            if row.get("base_failure_type") == "rare_case"
            and row.get("run_failure_type") != "rare_case"
        ]
        filtered.sort(key=lambda row: row["AP_delta_vs_base"], reverse=True)
    return filtered[:top_k]


def attribute_group_summary(rows: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        attr = row["attribute"]
        grouped[attr][row["status_vs_base"]] += 1
        grouped[attr][row["failure_transition"]] += 1
    return grouped


def focus_on_base_failure_set(base_summary: Dict, rows: Sequence[Dict], *, top_k: int) -> List[Dict]:
    keys = {
        (row["attribute"], row["value"])
        for row in failure_rows(base_summary)[:top_k]
    }
    focused = [row for row in rows if (row["attribute"], row["value"]) in keys]
    focused.sort(
        key=lambda row: (
            row["base_AP"] is None,
            row["base_AP"] if row["base_AP"] is not None else 0.0,
            row["attribute"],
            row["value"],
        )
    )
    return focused


def summarize_focus_rows(rows: Sequence[Dict]) -> Dict[str, int]:
    summary = {
        "compared_count": len(rows),
        "improved_count": 0,
        "declined_count": 0,
        "same_count": 0,
        "resolved_failure_count": 0,
        "still_failure_count": 0,
        "rare_to_not_rare_count": 0,
    }
    for row in rows:
        status = row.get("status_vs_base")
        if status == "improved":
            summary["improved_count"] += 1
        elif status == "declined":
            summary["declined_count"] += 1
        elif status == "same":
            summary["same_count"] += 1

        transition = row.get("failure_transition")
        if transition == "resolved_failure":
            summary["resolved_failure_count"] += 1
        elif transition == "still_failure":
            summary["still_failure_count"] += 1

        if row.get("base_failure_type") == "rare_case" and row.get("run_failure_type") != "rare_case":
            summary["rare_to_not_rare_count"] += 1
    return summary


def write_global_summary_csv(rows: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_global_summary_md(rows: Sequence[Dict], output_path: Path, *, split: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Base Failure-Set Compare Summary",
        "",
        f"- Split: `{split}`",
        f"- Compared base-selected failure cases per run: `{rows[0]['compared_count'] if rows else 0}`",
        "",
        "| Run | Overall AP Delta | Improved | Declined | Same | Rare -> Not Rare | Resolved Failure | Still Failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {format_float(row.get('overall_ap_delta'))} | "
            f"{row['improved_count']} | {row['declined_count']} | {row['same_count']} | "
            f"{row['rare_to_not_rare_count']} | {row['resolved_failure_count']} | {row['still_failure_count']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compare_md(
    *,
    base_summary: Dict,
    run_summary: Dict,
    rows: Sequence[Dict],
    output_path: Path,
    run_dir: Path,
    split: str,
    top_k: int,
) -> None:
    base_overall = base_summary.get("overall") or {}
    run_overall = run_summary.get("overall") or {}
    base_ap = base_overall.get("AP")
    run_ap = run_overall.get("AP")
    overall_delta = None if base_ap is None or run_ap is None else run_ap - base_ap
    grouped = attribute_group_summary(rows)
    counts = summarize_focus_rows(rows)

    lines = [
        "# Base Failure-Set Comparison",
        "",
        f"- Run: `{ensure_repo_relative(run_dir)}`",
        f"- Split: `{split}`",
        f"- Base predictions: `{base_summary.get('predictions_json', 'NA')}`",
        f"- Run predictions: `{run_summary.get('predictions_json', 'NA')}`",
        f"- Overall AP: `{format_float(base_ap)}` -> `{format_float(run_ap)}`",
        f"- Overall AP delta vs base: `{format_float(overall_delta)}`",
        f"- Compared base-selected failure cases: `{counts['compared_count']}`",
        "",
        "## Overall Counts",
        "",
        f"- Improved vs base: `{counts['improved_count']}`",
        f"- Declined vs base: `{counts['declined_count']}`",
        f"- Same as base: `{counts['same_count']}`",
        f"- Resolved failures: `{counts['resolved_failure_count']}`",
        f"- Still failures: `{counts['still_failure_count']}`",
        f"- Rare-case -> not-rare-case: `{counts['rare_to_not_rare_count']}`",
        "",
        "## Attribute Group Summary",
        "",
        "| Attribute | Improved | Declined | Same | Resolved Failure | New Failure | Still Failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for attribute in sorted(grouped):
        summary = grouped[attribute]
        lines.append(
            f"| `{attribute}` | {summary.get('improved', 0)} | {summary.get('declined', 0)} | "
            f"{summary.get('same', 0)} | {summary.get('resolved_failure', 0)} | "
            f"{summary.get('new_failure', 0)} | {summary.get('still_failure', 0)} |"
        )

    sections = [
        ("Improved Base Failure Cases", top_rows(rows, kind="improved", top_k=top_k)),
        ("Declined Base Failure Cases", top_rows(rows, kind="declined", top_k=top_k)),
        ("Base Rare Cases That Are No Longer Rare", top_rows(rows, kind="rare_to_not_rare", top_k=top_k)),
        ("Resolved Base Failure Cases", top_rows(rows, kind="resolved", top_k=top_k)),
        ("Still-Failing Base Cases", [row for row in rows if row["failure_transition"] == "still_failure"][:top_k]),
    ]

    for title, section_rows in sections:
        lines.extend(["", f"## {title}", ""])
        if not section_rows:
            lines.append("None.")
            continue
        for idx, row in enumerate(section_rows, start=1):
            lines.append(
                (
                    f"{idx}. `{row['attribute']}={row['value']}` | "
                    f"base_AP={format_float(row.get('base_AP'))} | "
                    f"run_AP={format_float(row.get('run_AP'))} | "
                    f"delta={format_float(row.get('AP_delta_vs_base'))} | "
                    f"status={row.get('status_vs_base')} | "
                    f"transition={row.get('failure_transition')}"
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_base_reports(args: argparse.Namespace) -> None:
    for split in args.splits:
        base_json = ensure_slice_stats(run_dir=args.base_run_dir, split=split, args=args)
        summary = load_json(base_json)
        failure_report_path = split_paths(args.base_run_dir, args.analysis_tag, split)["failure_md"]
        write_failure_report(
            summary,
            failure_report_path,
            top_k=args.top_k,
            run_dir=args.base_run_dir,
            split=split,
        )


def process_run(
    run_dir: Path,
    args: argparse.Namespace,
    base_summaries: Dict[str, Dict],
    global_summary_rows: Dict[str, List[Dict]],
) -> None:
    for split in args.splits:
        json_path = ensure_slice_stats(run_dir=run_dir, split=split, args=args)
        run_summary = load_json(json_path)
        paths = split_paths(run_dir, args.analysis_tag, split)
        focused_rows = focus_on_base_failure_set(
            base_summaries[split],
            compare_rows(base_summaries[split], run_summary),
            top_k=args.top_k,
        )
        write_compare_csv(focused_rows, paths["compare_csv"])
        write_compare_md(
            base_summary=base_summaries[split],
            run_summary=run_summary,
            rows=focused_rows,
            output_path=paths["compare_md"],
            run_dir=run_dir,
            split=split,
            top_k=args.top_k,
        )
        base_ap = (base_summaries[split].get("overall") or {}).get("AP")
        run_ap = (run_summary.get("overall") or {}).get("AP")
        counts = summarize_focus_rows(focused_rows)
        global_summary_rows[split].append(
            {
                "run": ensure_repo_relative(run_dir),
                "split": split,
                "overall_ap_base": "" if base_ap is None else round(base_ap, 6),
                "overall_ap_run": "" if run_ap is None else round(run_ap, 6),
                "overall_ap_delta": "" if base_ap is None or run_ap is None else round(run_ap - base_ap, 6),
                **counts,
            }
        )


def main() -> None:
    args = parse_args()
    args.splits = tuple(args.split or ["test"])

    args.base_run_dir = args.base_run_dir.resolve()
    args.base_log_file = args.base_log_file.resolve()
    args.copy_base_log_dir = args.copy_base_log_dir.resolve()
    args.report_md = args.report_md.resolve()
    args.attribute_file = args.attribute_file.resolve()
    args.train_attribute_file = args.train_attribute_file.resolve()
    args.corpus = args.corpus.resolve()

    if not args.base_run_dir.exists():
        raise FileNotFoundError(f"Missing base run directory: {args.base_run_dir}")
    if not args.base_log_file.exists():
        raise FileNotFoundError(f"Missing base log file: {args.base_log_file}")

    copied_base_log = copy_base_log(args.base_log_file, args.copy_base_log_dir)
    print(f"Copied base log to {ensure_repo_relative(copied_base_log)}")

    ensure_base_reports(args)
    base_summaries = {
        split: load_json(split_paths(args.base_run_dir, args.analysis_tag, split)["json"])
        for split in args.splits
    }

    run_dirs = discover_run_dirs(args)
    print(f"Discovered {len(run_dirs)} canonical run directories.")
    global_summary_rows: Dict[str, List[Dict]] = defaultdict(list)

    for idx, run_dir in enumerate(run_dirs, start=1):
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        print(f"[{idx}/{len(run_dirs)}] Processing {ensure_repo_relative(run_dir)}")
        process_run(run_dir, args, base_summaries, global_summary_rows)

    for split in args.splits:
        summary_rows = global_summary_rows[split]
        summary_rows.sort(
            key=lambda row: (
                row["overall_ap_delta"] == "",
                row["overall_ap_delta"] if row["overall_ap_delta"] != "" else 0.0,
            ),
            reverse=True,
        )
        write_global_summary_csv(
            summary_rows,
            args.copy_base_log_dir / f"{split}_base_failure_set_compare_summary.csv",
        )
        write_global_summary_md(
            summary_rows,
            args.copy_base_log_dir / f"{split}_base_failure_set_compare_summary.md",
            split=split,
        )

    print("Finished generating posthoc failure-case and base-compare reports.")


if __name__ == "__main__":
    main()
