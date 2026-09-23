# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair the two missing current-corpus PAIR/TAP Mixtral source cases in FDR."""

from __future__ import annotations

import os
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
APP_NAME = "harmbench-pair-tap-precomputed-repair"
MODEL_VOLUME_NAME = "harmbench-mixtral-attacker-cache"
OUTPUT_VOLUME_NAME = "harmbench-pair-tap-precomputed-repair-results"
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
EXPERIMENT = "mixtral_8x7b"
SOURCE_HASHES = {
    "PAIR": "9a85a36ae7aa47fef160eca99746379a1521f75acb2d85749ed083e0aac4af29",  # pragma: allowlist secret
    "TAP": "5b4dd35089a4b02458551b3cfbb6b747787a7bf14012775ef1d441d6aaf24867",  # pragma: allowlist secret
}

model_cache = modal.Volume.from_name(MODEL_VOLUME_NAME, environment_name="FDR", create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME_NAME, environment_name="FDR", create_if_missing=True)
base_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.21.0",
        setup_dockerfile_commands=[
            "RUN command -v python3 && python3 --version && "
            "(command -v python || ln -s /usr/bin/python3 /usr/local/bin/python) && "
            "(command -v pip || ln -s /usr/bin/pip3 /usr/local/bin/pip)"
        ],
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.12.0",
        "anthropic",
        "art",
        "boto3",
        "bpe",
        "datasketch==2.0.0",
        "fschat",
        "google-cloud-aiplatform",
        "google-generativeai",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/"
        "en_core_web_sm-3.7.1-py3-none-any.whl",
        "matplotlib",
        "mistralai",
        "numpy==1.26.4",
        "openai>=1.25.1",
        "pandas<3",
        "pyyaml",
        "ray",
        "sentence-transformers",
        "spacy==3.7.2",
        "tenacity",
        "transformers==4.57.1",
    )
    .run_commands(
        "git init /app/HarmBench && "
        "git -C /app/HarmBench remote add origin https://github.com/centerforaisafety/HarmBench.git && "
        f"git -C /app/HarmBench fetch --depth 1 origin {UPSTREAM_REVISION} && "
        "git -C /app/HarmBench checkout --detach FETCH_HEAD"
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
if modal.is_local():
    source_csv = Path(os.environ["HARMBENCH_SOURCE_CSV"]).resolve(strict=True)
    pair_cases = Path(os.environ["HARMBENCH_PAIR_PRECOMPUTED"]).resolve(strict=True)
    tap_cases = Path(os.environ["HARMBENCH_TAP_PRECOMPUTED"]).resolve(strict=True)
    image = (
        base_image.add_local_file(source_csv, "/app/behaviors.csv")
        .add_local_file(pair_cases, "/app/pair-precomputed.json")
        .add_local_file(tap_cases, "/app/tap-precomputed.json")
    )
else:
    image = base_image

app = modal.App(APP_NAME)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=image,
    gpu="H200:2",
    cpu=16,
    memory=262144,
    timeout=86400,
    startup_timeout=1800,
    min_containers=0,
    max_containers=2,
    volumes={"/root/.cache/huggingface": model_cache, "/outputs": outputs},
)
def repair(method: str, run_id: str) -> dict:
    import csv
    import json
    import re
    import subprocess
    import sys

    if method not in {"PAIR", "TAP"}:
        raise ValueError("method must be PAIR or TAP")
    if not re.fullmatch(r"[a-z0-9-]+", run_id):
        raise ValueError("run_id must contain lowercase letters, digits, or hyphens")
    source = Path("/app/behaviors.csv")
    official = Path(f"/app/{method.lower()}-precomputed.json")
    if _sha256(official) != SOURCE_HASHES[method]:
        raise ValueError("precomputed PAIR/TAP source hash changed")
    with source.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    official_cases = json.loads(official.read_text(encoding="utf-8"))
    missing_rows = [row for row in all_rows if row["BehaviorID"] not in official_cases]
    if len(all_rows) != 400 or len(missing_rows) != 2:
        raise ValueError("PAIR/TAP repair must cover exactly two missing current-corpus behaviors")
    root = Path("/outputs") / run_id / method.lower()
    receipt_path = root / "repair-receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "completed" or receipt.get("method") != method:
            raise ValueError("existing PAIR/TAP repair receipt is invalid")
        return {
            "method": method,
            "run_id": run_id,
            "status": "completed",
            "repaired_behaviors": receipt["repaired_behaviors"],
            "output_volume": OUTPUT_VOLUME_NAME,
        }
    root.mkdir(parents=True, exist_ok=True)
    missing_csv = root / "missing-behaviors.csv"
    with missing_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(missing_rows)
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    upstream = Path("/app/HarmBench")
    diagnostic_path = root / "upstream-diagnostic.log"
    completed = subprocess.run(
        [
            sys.executable,
            "generate_test_cases.py",
            "--method_name",
            method,
            "--experiment_name",
            EXPERIMENT,
            "--behaviors_path",
            str(missing_csv),
            "--save_dir",
            str(generated),
        ],
        cwd=upstream,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        diagnostic_path.write_text(completed.stdout, encoding="utf-8")
        outputs.commit()
        safe_lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith(
                ("ModuleNotFoundError:", "ImportError:", "KeyError:", "ValueError:", "TypeError:", "RuntimeError:")
            )
        ]
        safe_error = safe_lines[-1] if safe_lines else f"exit_status_{completed.returncode}"
        raise RuntimeError(f"upstream {method} repair failed: {safe_error[:400]}")
    cases_path = generated / "test_cases.json"
    if not cases_path.is_file():
        subprocess.run(
            [sys.executable, "merge_test_cases.py", "--method_name", method, "--save_dir", str(generated)],
            cwd=upstream,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    repaired = json.loads(cases_path.read_text(encoding="utf-8"))
    if set(repaired) != {row["BehaviorID"] for row in missing_rows} or any(
        not isinstance(value, list) or len(value) != 1 for value in repaired.values()
    ):
        raise ValueError("PAIR/TAP repair output does not cover exactly the two missing behaviors")
    config_path = upstream / f"configs/method_configs/{method}_config.yaml"
    models_path = upstream / "configs/model_configs/models.yaml"
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "method": method,
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "upstream_revision": UPSTREAM_REVISION,
        "source_precomputed_sha256": SOURCE_HASHES[method],
        "missing_behaviors_sha256": _sha256(missing_csv),
        "repaired_cases_sha256": _sha256(cases_path),
        "method_config_sha256": _sha256(config_path),
        "models_config_sha256": _sha256(models_path),
        "repaired_behaviors": len(repaired),
        "repaired_cases": sum(len(value) for value in repaired.values()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_cache.commit()
    outputs.commit()
    return {
        "method": method,
        "run_id": run_id,
        "status": "completed",
        "repaired_behaviors": receipt["repaired_behaviors"],
        "repaired_cases": receipt["repaired_cases"],
        "output_volume": OUTPUT_VOLUME_NAME,
    }


@app.local_entrypoint()
def main(run_id: str) -> None:
    import json

    calls = []
    for method in ("PAIR", "TAP"):
        call = repair.spawn(method, run_id)
        calls.append({"method": method, "function_call_id": call.object_id})
    print(json.dumps({"run_id": run_id, "calls": calls}, sort_keys=True))
