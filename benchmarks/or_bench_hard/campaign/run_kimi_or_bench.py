"""Run OR-Bench through Gym without persisting credentials."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from yaml import safe_load


ROOT = Path(__file__).parent
WORKSPACE = Path("/Users/kruge/Documents/ChatGPT/NVIDIA")
MODEL_CONFIG = Path("/tmp/agentdyn-kimi-k3.yaml")


def main() -> None:
    values = dotenv_values(WORKSPACE / ".env")
    model = safe_load(MODEL_CONFIG.read_text())["policy_model"]["responses_api_models"]["inference_provider"]
    token = values.get("MODAL_PROXY_TOKEN")
    if not token:
        raise RuntimeError("MODAL_PROXY_TOKEN is required in the workspace .env")
    environment = os.environ | {
        "OR_BENCH_POLICY_BASE_URL": model["base_url"],
        "OR_BENCH_POLICY_API_KEY": token,
        "OR_BENCH_POLICY_MODEL": model["model"],
        "OR_BENCH_JUDGE_BASE_URL": "https://snorkelai-fdr--ep-glm-5-3-flash-server.us-west.modal.direct/v1",
        "OR_BENCH_JUDGE_API_KEY": token,
        "OR_BENCH_JUDGE_MODEL": "zai-org/GLM-5.3-Flash",
    }
    output = ROOT / "results" / "or-bench-hard-20260921" / "kimi-valid-cap.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ".venv/bin/gym",
            "eval",
            "run",
            "--benchmark",
            "or_bench_hard",
            "--config",
            str(ROOT / ".local-or-bench-run.yaml"),
            "--output",
            str(output),
            "--num-repeats",
            "1",
            "--split",
            "benchmark",
            "--limit",
            os.environ.get("OR_BENCH_LIMIT", "5"),
            "--concurrency",
            os.environ.get("OR_BENCH_CONCURRENCY", "2"),
            "--max-output-tokens",
            os.environ.get("OR_BENCH_POLICY_MAX_OUTPUT_TOKENS", "8192"),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


if __name__ == "__main__":
    main()
