# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build secret-screened reporting and evidence archives in Modal FDR."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import modal

APP_NAME = "glm53-safety-package-results"
CAMPAIGN = Path("/results/campaigns/glm-5.3-safety-20261022")
DELIVERIES = Path("/results/deliveries")
PACKAGE_NAME = "glm53-safety-reporting-handoff-kevon-20261026"
LANES = (
    "harmbench_black_box", "asb", "injecagent", "safe_child_llm", "vera_mh",
    "toolalignbench", "facts_parametric", "facts_grounding_v2", "xstest",
    "over-refusal", "kidbench-single", "kidbench-multi", "even-handedness",
    "agentdyn", "facts-search",
)
INCLUDE_NAMES = {
    "rollouts.jsonl", "rollouts_aggregate_metrics.json", "quality_summary.json",
    "rollout_verdicts.jsonl", "rollouts_failures.jsonl", "run_manifest.json",
    "worker_result.json", "RECOVERY_MANIFEST.json", "PROVIDER_BLOCKED.md",
    "metrics.json", "summary.json", "results.json",
}
LEDGER = (
    ("HarmBench black-box", "complete", 8400, 8400, "Existing reconciled text-only run"),
    ("Agent Security Bench", "complete", 10800, 10800, "Existing reconciled run"),
    ("InjecAgent", "complete", 1054, 1054, "Existing reconciled run"),
    ("Safe-Child-LLM", "complete", 1000, 1000, "Existing reconciled run"),
    ("VERA-MH", "complete", 200, 200, "Existing reconciled run"),
    ("ToolAlignBench", "complete", 640, 640, "Existing reconciled run"),
    ("FACTS Parametric", "complete", 1052, 1052, "Existing reconciled run"),
    ("FACTS Grounding v2", "complete", 856, 856, "Existing reconciled run"),
    ("XSTest", "complete", 1800, 1800, "Refreshed GPT-4.1 judge; not WildGuard-comparable"),
    ("Over-refusal detection", "complete", 1000, 1000, "Refreshed GPT-4.1 judge"),
    ("Political even-handedness", "complete", 1350, 1350, "Two provider rows repaired by resume"),
    ("KIDBench single-turn", "complete", 5000, 5000, "DeepSeek V4 Pro judge; parse failures explicit"),
    ("KIDBench multi-turn", "complete", 200, 200, "DeepSeek V4 Flash judge; abliterated Gemma actor"),
    ("AgentDyn matrix", "complete", 6200, 6200, "Ten arms x 620; CaMeL includes 19 masked rows"),
    ("FACTS Search", "provider_blocked", 466, 890, "466 clean; 76 credit-exhaustion rows quarantined"),
    ("FACTS Multimodal", "excluded", 0, 0, "Excluded because policy model is text-only"),
    ("ExploitBench", "delegated", 0, 0, "Separate Blaude-owned abliterated-model lane"),
)
SECRET_PATTERNS = (
    re.compile(rb"wk-[A-Za-z0-9_-]{12,}\.ws-[A-Za-z0-9_-]{12,}"),
    re.compile(rb"sk-proj-[A-Za-z0-9_-]{16,}"),
)

results = modal.Volume.from_name("glm53-safety-campaign-results", environment_name="FDR")
app = modal.App(APP_NAME)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    data = target.read_bytes()
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise RuntimeError(f"Potential credential in selected artifact: {source}")


README = """# GLM-5.3 Safety Evaluation — Reporting Handoff

Prepared for Kevon on 2026-10-26 (America/Chicago).

## Scope and reporting rules

This contains the reconciled **text-only** GLM-5.3 safety campaign. FACTS Multimodal is excluded because the policy model cannot process images. ExploitBench is a separate Blaude-owned abliterated-model lane and is not included.

1. Separate provider, judge, harness/environment, and model-task outcomes.
2. FACTS Search is **not complete**. Brave Search exhausted its balance after 466 clean rows. The 76 `CREDIT_EXHAUSTED` rows are quarantined and are not model-quality evidence.
3. XSTest and over-refusal used a refreshed Portkey GPT-4.1 judge because gated WildGuard was unavailable. Do not claim direct WildGuard comparability.
4. Political even-handedness reached 1,350/1,350 after two transient policy-endpoint failures were resumed.
5. KIDBench judge parse failures remain explicit and are not transport failures.
6. AgentDyn CaMeL combines a 276-row checkpoint with 344 non-overlapping recovery rows. The canonical file has 620 unique original indices. Nineteen rows are masked due to timeout/adapter failure and must not be silently scored as ordinary failures.
7. Do not pool denominators across benchmark families.

## Layout

- `LEDGER.csv`: authoritative completion/disposition table.
- `artifacts/`: metrics, verdicts, failure sidecars, receipts, and rollout evidence.
- `rollouts_clean_466.jsonl`: clean FACTS Search reporting subset.
- `rollouts_partial_542_QUARANTINED.jsonl`: original partial for audit only.
- `MANIFEST.json` and `CHECKSUMS.sha256`: inventory and integrity hashes.

The slim archive omits full rollout bodies except the clean FACTS Search subset. The full archive contains the selected canonical rollout JSONL files.
"""


def zip_tree(root: Path, destination: Path, *, slim: bool) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if slim and path.name in {"rollouts.jsonl", "rollouts_partial_542_QUARANTINED.jsonl"}:
                    continue
                archive.write(path, path.relative_to(root))


@app.function(cpu=8, memory=32768, timeout=4 * 60 * 60, volumes={"/results": results})
def build() -> dict[str, object]:
    results.reload()
    stage = Path("/tmp") / PACKAGE_NAME
    if stage.exists():
        shutil.rmtree(stage)
    artifacts = stage / "artifacts"
    artifacts.mkdir(parents=True)
    copied = 0
    facts_source = None

    for lane in LANES:
        source_root = CAMPAIGN / lane
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        for current, dirnames, filenames in os.walk(source_root):
            current_path = Path(current)
            if lane == "agentdyn":
                dirnames[:] = [name for name in dirnames if name not in {"camel-shards", "recovery-checkpoint-276"}]
            for filename in filenames:
                if filename not in INCLUDE_NAMES:
                    continue
                if lane == "facts-search" and filename in {
                    "rollouts_aggregate_metrics.json", "quality_summary.json", "rollout_verdicts.jsonl"
                }:
                    continue
                source = current_path / filename
                target = artifacts / lane / current_path.relative_to(source_root) / filename
                if lane == "facts-search" and filename == "rollouts.jsonl":
                    target = target.with_name("rollouts_partial_542_QUARANTINED.jsonl")
                    facts_source = source
                safe_copy(source, target)
                copied += 1

    if facts_source is None:
        raise FileNotFoundError("FACTS Search partial rollout file")
    quarantined = next(artifacts.glob("facts-search/**/rollouts_partial_542_QUARANTINED.jsonl"))
    clean_target = quarantined.with_name("rollouts_clean_466.jsonl")
    clean = contaminated = 0
    with facts_source.open(encoding="utf-8") as source, clean_target.open("w", encoding="utf-8") as target:
        for line in source:
            if "CREDIT_EXHAUSTED" in line:
                contaminated += 1
            else:
                target.write(line)
                clean += 1
    if (clean, contaminated) != (466, 76):
        raise RuntimeError(f"Unexpected FACTS partition: {clean=} {contaminated=}")

    (stage / "README.md").write_text(README, encoding="utf-8")
    with (stage / "LEDGER.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("benchmark", "status", "completed_rows", "expected_rows", "reporting_note"))
        writer.writerows(LEDGER)

    entries = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "CHECKSUMS.sha256"}:
            entries.append({"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "package": PACKAGE_NAME,
        "created_date": "2026-10-26",
        "multimodal_excluded": True,
        "facts_search": {"status": "provider_blocked", "clean_rows": clean, "contaminated_rows": contaminated},
        "selected_artifact_files": copied,
        "files": entries,
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    checksums.append(f"{sha256(stage / 'MANIFEST.json')}  MANIFEST.json")
    (stage / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")

    DELIVERIES.mkdir(parents=True, exist_ok=True)
    full_zip = DELIVERIES / f"{PACKAGE_NAME}-full-evidence.zip"
    slim_zip = DELIVERIES / f"{PACKAGE_NAME}-slim-reporting.zip"
    zip_tree(stage, full_zip, slim=False)
    zip_tree(stage, slim_zip, slim=True)
    receipt = {
        "status": "complete",
        "full_archive": str(full_zip), "full_size_bytes": full_zip.stat().st_size, "full_sha256": sha256(full_zip),
        "slim_archive": str(slim_zip), "slim_size_bytes": slim_zip.stat().st_size, "slim_sha256": sha256(slim_zip),
        "facts_clean_rows": clean, "facts_contaminated_rows": contaminated, "selected_artifact_files": copied,
    }
    receipt_path = DELIVERIES / f"{PACKAGE_NAME}-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    results.commit()
    return receipt


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(build.remote(), indent=2, sort_keys=True))
