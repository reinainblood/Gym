# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modal FDR deployment for the collaborative Safe-Child-LLM annotation app."""

from __future__ import annotations

import os
from pathlib import Path

import modal


ENVIRONMENT = "FDR"
APP_NAME = "safe-child-annotation"
VOLUME_NAME = "safe-child-annotation-data"
SECRET_NAME = "safe-child-annotation-auth"
DATA_DIR = Path("/data")
RUNS_DIR = DATA_DIR / "runs"
DATABASE_PATH = DATA_DIR / "annotations.sqlite3"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT, create_if_missing=True)
auth_secret = modal.Secret.from_name(
    SECRET_NAME,
    environment_name=ENVIRONMENT,
    required_keys=["SAFE_CHILD_ACCESS_CODE", "SAFE_CHILD_COOKIE_SECRET"],
)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi==0.117.1", "pydantic==2.11.9", "uvicorn==0.37.0")
    .add_local_python_source("benchmarks.safe_child_llm", "resources_servers.safe_child_llm")
)


def _run_specs() -> list[tuple[str, Path]]:
    if not RUNS_DIR.exists():
        return []
    return [(path.stem, path) for path in sorted(RUNS_DIR.glob("*.jsonl"))]


@app.function(
    image=image,
    volumes={str(DATA_DIR): volume},
    secrets=[auth_secret],
    min_containers=1,
    max_containers=1,
    scaledown_window=120,
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from benchmarks.safe_child_llm.collaborative_annotation_app import AnnotationDatabase, create_collaborative_app

    database = AnnotationDatabase(DATABASE_PATH)
    database.import_specs(_run_specs())
    volume.commit()

    def reload_data() -> dict[str, int]:
        volume.reload()
        return database.import_specs(_run_specs())

    return create_collaborative_app(
        database,
        access_code=os.environ["SAFE_CHILD_ACCESS_CODE"],
        cookie_secret=os.environ["SAFE_CHILD_COOKIE_SECRET"],
        persist=volume.commit,
        reload_data=reload_data,
    )
