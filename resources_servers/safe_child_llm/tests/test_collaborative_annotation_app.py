# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import shutil
import subprocess

from fastapi.testclient import TestClient

from benchmarks.safe_child_llm.collaborative_annotation_app import (
    APP_HTML,
    AnnotationDatabase,
    create_collaborative_app,
)


def _rollout(identifier: str, text: str) -> dict:
    return {
        "safe_child_id": identifier,
        "age_group": "6-12",
        "category": "test",
        "source": "fixture",
        "prompt": "A test prompt",
        "response": {
            "model": "endpoint-model",
            "output": [{"type": "message", "role": "assistant", "content": [{"text": text}]}],
        },
    }


def _login(client: TestClient, annotator: str = "Reviewer") -> None:
    response = client.post("/api/login", json={"access_code": "project-code", "annotator": annotator})
    assert response.status_code == 200


def test_import_four_models_and_annotation_progress(tmp_path):
    database = AnnotationDatabase(tmp_path / "annotations.sqlite3")
    specs = []
    for model in ("kimi-k3", "qwen-3.5", "super-3.5-vl", "ultra-3"):
        path = tmp_path / f"{model}.jsonl"
        path.write_text(json.dumps(_rollout("safe-child-6-12-001", f"{model} answer")) + "\n")
        specs.append((model, path))
    assert database.import_specs(specs) == {model: 1 for model, _ in specs}
    client = TestClient(
        create_collaborative_app(database, access_code="project-code", cookie_secret="cookie-secret"),
        base_url="https://testserver",
    )
    assert "Enter the shared project access code" in client.get("/").text
    assert client.get("/api/summary").status_code == 401
    assert client.post("/api/login", json={"access_code": "wrong", "annotator": "Reviewer"}).status_code == 401
    _login(client)
    assert "SAFE-CHILD / HUMAN REVIEW" in client.get("/").text
    summary = client.get("/api/summary").json()
    assert [row["model"] for row in summary["models"]] == ["kimi-k3", "qwen-3.5", "super-3.5-vl", "ultra-3"]
    assert summary["total"] == 4 and summary["labeled_by_annotator"] == 0
    item = client.get("/api/item?model=kimi-k3&pending_only=true").json()
    assert set(item["responses"]) == {"kimi-k3", "qwen-3.5", "super-3.5-vl", "ultra-3"}
    saved = client.post(
        "/api/annotations",
        json={
            "safe_child_id": "safe-child-6-12-001",
            "model": "kimi-k3",
            "harmful": 0,
            "action_label": 1,
            "notes": "Developmentally safe refusal.",
        },
    )
    assert saved.status_code == 200 and saved.json()["annotator"] == "Reviewer"
    assert client.get("/api/item?model=kimi-k3&pending_only=true").json()["empty"] is True
    assert client.get("/api/summary").json()["labeled_by_annotator"] == 1


def test_annotators_are_independent_and_exports_are_attributed(tmp_path):
    database = AnnotationDatabase(tmp_path / "annotations.sqlite3")
    path = tmp_path / "super.jsonl"
    path.write_text(json.dumps(_rollout("safe-child-6-12-001", "answer")) + "\n")
    database.import_jsonl(path, "super")
    app = create_collaborative_app(database, access_code="project-code", cookie_secret="cookie-secret")
    first = TestClient(app, base_url="https://testserver")
    second = TestClient(app, base_url="https://testserver")
    _login(first, "Kirsten")
    _login(second, "Coworker")
    payload = {
        "safe_child_id": "safe-child-6-12-001",
        "model": "super",
        "harmful": 0,
        "action_label": 0,
        "notes": "",
    }
    assert first.post("/api/annotations", json=payload).status_code == 200
    payload.update(harmful=1, action_label=4)
    assert second.post("/api/annotations", json=payload).status_code == 200
    assert first.get("/api/summary").json()["all_annotations"] == 2
    own = first.get("/api/export.jsonl").text
    all_rows = first.get("/api/export.jsonl?all_annotators=true").text
    assert "Kirsten" in own and "Coworker" not in own
    assert "Kirsten" in all_rows and "Coworker" in all_rows
    csv_text = first.get("/api/export.csv?all_annotators=true").text
    assert "action_label_name" in csv_text and "Total Compliance" in csv_text


def test_import_deduplicates_ids_for_a_model_and_keeps_last_row(tmp_path):
    database = AnnotationDatabase(tmp_path / "annotations.sqlite3")
    path = tmp_path / "duplicates.jsonl"
    first = json.dumps(_rollout("safe-child-6-12-001", "first answer"))
    last = json.dumps(_rollout("safe-child-6-12-001", "last answer"))
    path.write_text(first + "\n" + last + "\n")
    assert database.import_jsonl(path, "model") == 1
    with database._connect() as connection:
        row = connection.execute(
            "SELECT response_text FROM items WHERE safe_child_id=? AND model=?",
            ("safe-child-6-12-001", "model"),
        ).fetchone()
    assert row[0] == "last answer"


def test_browser_script_parses(tmp_path):
    if not shutil.which("node"):
        return
    script = re.search(r"<script>(.*)</script>", APP_HTML, re.DOTALL)
    assert script
    path = tmp_path / "app.js"
    path.write_text(script.group(1))
    subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)


def test_authenticated_reload_imports_new_runs(tmp_path):
    database = AnnotationDatabase(tmp_path / "annotations.sqlite3")
    client = TestClient(
        create_collaborative_app(
            database,
            access_code="project-code",
            cookie_secret="cookie-secret",
            reload_data=lambda: {"super": 1000},
        ),
        base_url="https://testserver",
    )
    assert client.post("/api/reload").status_code == 401
    _login(client)
    assert client.post("/api/reload").json()["imported"] == {"super": 1000}
