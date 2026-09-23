# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the KORA rows a NeMo Gym run consumes.

KORA publishes each leaderboard run as a self-contained package of Parquet tables at
https://korabench.ai/data. Besides the graded conversations it carries the instrument
itself: the 781 scenarios with their opening messages and (for relationship risks) the
memory injected into the target, the 26-risk taxonomy, the 7 behaviour rubrics, the tier
system prompts and the harness profile. This module:

1. downloads that package (pinned by URL and SHA-256 in :mod:`benchmarks.kora.upstream_spec`)
   into ``benchmarks/kora/upstream/`` (gitignored) and checks the per-table hashes in its
   ``manifest.json``;
2. exports the taxonomy and the behaviour rubrics to ``upstream/pack.json``, which the
   resources server reads at run time; and
3. writes one Gym row per (scenario, tier): 781 scenarios x {adult, child} = 1,562 rows,
   each carrying the target's system prompt in ``responses_create_params.input`` and the
   scenario fields the child simulator and the judge need.

Nothing from the package is committed except the five ``example.jsonl`` rows. KORA's
Permitted Use terms (https://korabench.ai/terms) allow research, internal evaluation and
benchmarking with attribution; the full corpus is fetched, not redistributed.

Usage::

    python -m benchmarks.kora.prepare                      # download, verify, write all rows
    python -m benchmarks.kora.prepare --tier child         # one tier only
    python -m benchmarks.kora.prepare --package-zip ~/Downloads/kora-run-341620448e7e1486.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

from benchmarks.kora.upstream_spec import (
    AGE_RANGES,
    EXPECTED_BEHAVIORS,
    EXPECTED_RISKS,
    EXPECTED_ROWS,
    EXPECTED_SCENARIOS,
    PACKAGE_DATA_HASH,
    PACKAGE_DIRNAME,
    PACKAGE_FILENAME,
    PACKAGE_FORMAT_VERSION,
    PACKAGE_SCHEMA_HASH,
    PACKAGE_SHA256,
    PACKAGE_URL,
    RUN_ID,
    TIERS,
    UPSTREAM_REVISION,
    build_target_system_prompt,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "benchmarks" / "kora" / "upstream"
DEFAULT_PACK_FPATH = DEFAULT_UPSTREAM_DIR / "pack.json"
OUTPUT_FPATH = REPO_ROOT / "benchmarks" / "kora" / "data" / "kora_benchmark.jsonl"
DEFAULT_SERVER_DATA_DIR = REPO_ROOT / "resources_servers" / "kora" / "data"

#: Tables the adapter reads. Their ``manifest.json`` hashes are checked after unpacking.
REQUIRED_TABLES = ("scenarios", "risks", "behaviors", "leaderboard", "results")

#: The committed smoke sample: one scenario per risk below, alternating tiers. These five
#: risks were chosen for the least sensitive narratives in the corpus; every other row is
#: fetched, never committed.
EXAMPLE_RISKS = (
    ("academic_dishonesty_and_misuse", "child"),
    ("factual_hallucinations_and_misinformation", "adult"),
    ("cognitive_atrophy_and_dependency", "child"),
    ("sensorimotor_displacement", "adult"),
    ("privacy_and_personal_data_protection", "child"),
)


class UpstreamError(RuntimeError):
    """The package is missing, fails its checksums, or does not describe the pinned run."""


# ---------------------------------------------------------------------------
# Package acquisition
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_package(zip_path: Path) -> Path:
    """Fetch the run package from korabench.ai into ``zip_path``; the caller verifies it."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (about 280 MB) to %s", PACKAGE_URL, zip_path)
    with tempfile.NamedTemporaryFile(dir=zip_path.parent, delete=False) as tmp:
        with urllib.request.urlopen(PACKAGE_URL, timeout=120) as response:  # noqa: S310 - pinned https URL
            shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(zip_path)
    return zip_path


def _manifest_of(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise UpstreamError(f"No manifest.json under {package_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_package_dir(package_dir: Path) -> dict[str, Any]:
    """Check the unpacked package is the pinned run and its required tables are intact."""
    manifest = _manifest_of(package_dir)
    identity = (manifest.get("run_id"), manifest.get("format_version"), manifest.get("schema_hash"))
    expected = (RUN_ID, PACKAGE_FORMAT_VERSION, PACKAGE_SCHEMA_HASH)
    if identity != expected:
        raise UpstreamError(f"Package identity {identity} != pinned {expected} at {package_dir}")
    if manifest.get("data_hash") != PACKAGE_DATA_HASH:
        raise UpstreamError(f"Package data_hash {manifest.get('data_hash')} != pinned {PACKAGE_DATA_HASH}")
    tables = {table["name"]: table for table in manifest.get("tables", [])}
    for name in REQUIRED_TABLES:
        table = tables.get(name)
        if table is None:
            raise UpstreamError(f"manifest.json lists no table {name!r}")
        path = package_dir / table["file"]
        if not path.exists():
            raise UpstreamError(f"Table file missing: {path}")
        actual = sha256_of(path)
        if actual != table["sha256"]:
            raise UpstreamError(f"{path.name}: sha256 {actual} != manifest {table['sha256']}")
    return manifest


def ensure_package(
    upstream_dir: Path = DEFAULT_UPSTREAM_DIR,
    *,
    zip_path: Optional[Path] = None,
    download: bool = True,
) -> Path:
    """Return the unpacked, verified package directory, fetching and unpacking as needed.

    Idempotent: an existing directory that passes :func:`verify_package_dir` is left alone,
    so this is safe from a test's ``pytest_configure``.
    """
    package_dir = upstream_dir / PACKAGE_DIRNAME
    if (package_dir / "manifest.json").exists():
        verify_package_dir(package_dir)
        return package_dir

    zip_path = zip_path or (upstream_dir / PACKAGE_FILENAME)
    if not zip_path.exists():
        if not download:
            raise UpstreamError(f"No package at {zip_path} and downloading is disabled")
        download_package(zip_path)
    actual = sha256_of(zip_path)
    if actual != PACKAGE_SHA256:
        raise UpstreamError(f"{zip_path}: sha256 {actual} != pinned {PACKAGE_SHA256}")

    logger.info("Unpacking %s into %s", zip_path, upstream_dir)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not all(name.startswith(PACKAGE_DIRNAME + "/") for name in names):
            raise UpstreamError(f"{zip_path} does not unpack into a single {PACKAGE_DIRNAME}/ directory")
        archive.extractall(upstream_dir)
    verify_package_dir(package_dir)
    return package_dir


def resolve_package(upstream_dir: Path = DEFAULT_UPSTREAM_DIR) -> Path:
    """A verified package directory, without network access; raises when it is absent."""
    package_dir = upstream_dir / PACKAGE_DIRNAME
    if not (package_dir / "manifest.json").exists():
        raise UpstreamError(f"No KORA package under {upstream_dir}. Run `python -m benchmarks.kora.prepare` first.")
    verify_package_dir(package_dir)
    return package_dir


# ---------------------------------------------------------------------------
# Pack export (taxonomy + behaviour rubrics)
# ---------------------------------------------------------------------------


def _read_table(package_dir: Path, name: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(package_dir / f"{name}.parquet")
    rows = table.to_pylist()
    return [
        {key: (None if value != value else value) if isinstance(value, float) else value for key, value in row.items()}
        for row in rows
    ]


def build_pack(package_dir: Path) -> dict[str, Any]:
    """The run's taxonomy and behaviour set, in the field names the verifier uses."""
    behaviors = [
        {
            "id": row["behavior_id"],
            "code": row["code"] or row["behavior_id"],
            "name": row["name"],
            "level": row["level"],
            "precondition": row["precondition"] or None,
            "assessment_prompt": row["assessment_prompt"],
        }
        for row in _read_table(package_dir, "behaviors")
    ]
    risks = [
        {
            "risk_id": row["risk_id"],
            "risk_category_id": row["risk_category_id"],
            "risk_category_name": row["risk_category_name"],
            "name": row["name"],
            "description": row["description"],
            "planned_turns": int(row["planned_turns"]),
            "provide_user_context": bool(row["provide_user_context"]),
        }
        for row in _read_table(package_dir, "risks")
    ]
    if len(behaviors) != EXPECTED_BEHAVIORS:
        raise UpstreamError(f"Expected {EXPECTED_BEHAVIORS} behaviours, package has {len(behaviors)}")
    if len(risks) != EXPECTED_RISKS:
        raise UpstreamError(f"Expected {EXPECTED_RISKS} risks, package has {len(risks)}")
    return {
        "run_id": RUN_ID,
        "format_version": PACKAGE_FORMAT_VERSION,
        "upstream_revision": UPSTREAM_REVISION,
        "behaviors": behaviors,
        "risks": risks,
    }


def write_pack(package_dir: Path, pack_fpath: Path = DEFAULT_PACK_FPATH) -> Path:
    pack_fpath.parent.mkdir(parents=True, exist_ok=True)
    pack_fpath.write_text(json.dumps(build_pack(package_dir), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pack_fpath


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _scenario_rows(package_dir: Path) -> list[dict[str, Any]]:
    scenarios = sorted(_read_table(package_dir, "scenarios"), key=lambda row: row["scenario_id"])
    if len(scenarios) != EXPECTED_SCENARIOS:
        raise UpstreamError(f"Expected {EXPECTED_SCENARIOS} scenarios, package has {len(scenarios)}")
    return scenarios


def build_row(scenario: dict[str, Any], risk: dict[str, Any], *, tier: str) -> dict[str, Any]:
    """One Gym row: the target's system prompt plus everything the harness and judge read.

    ``responses_create_params.input`` holds only the system message. The child's turns are
    produced at run time by the ``kora_child_actor`` agent, starting from the scenario's
    stored ``first_user_message`` exactly as upstream does.
    """
    if scenario["age_range"] not in AGE_RANGES:
        raise UpstreamError(f"{scenario['scenario_id']}: unexpected age_range {scenario['age_range']!r}")
    memory = scenario.get("model_memory") or None
    if bool(memory) != risk["provide_user_context"]:
        raise UpstreamError(
            f"{scenario['scenario_id']}: model_memory {'present' if memory else 'absent'} but risk "
            f"{risk['risk_id']} provide_user_context={risk['provide_user_context']}"
        )
    system_prompt = build_target_system_prompt(tier=tier, age_range=scenario["age_range"], model_memory=memory)
    return {
        "kora_id": f"{scenario['scenario_id']}:{tier}",
        "run_id": RUN_ID,
        "scenario_id": scenario["scenario_id"],
        "tier": tier,
        "age_range": scenario["age_range"],
        "risk_category_id": risk["risk_category_id"],
        "risk_category_name": risk["risk_category_name"],
        "risk_id": risk["risk_id"],
        "risk_name": risk["name"],
        "risk_description": risk["description"],
        "planned_turns": int(scenario["planned_turns"]),
        "scenario_flavor_id": scenario.get("scenario_flavor_id"),
        "short_title": scenario["short_title"],
        "child_age": int(scenario["child_age"]),
        "child_gender": scenario.get("child_gender"),
        "child_race_ethnicity": scenario.get("child_race_ethnicity"),
        "child_ses": scenario.get("child_ses"),
        "child_cognitive_maturity": scenario["child_cognitive_maturity"],
        "child_emotional_maturity": scenario["child_emotional_maturity"],
        "risk_signal_type": scenario.get("risk_signal_type"),
        "social_context": scenario.get("social_context"),
        "motivation_name": scenario.get("motivation_name"),
        "child_maturity": scenario["child_maturity"],
        "child_background": scenario["child_background"],
        "narrative": scenario["narrative"],
        "evaluation_criteria": scenario.get("evaluation_criteria"),
        "first_user_message": scenario["first_user_message"],
        "model_memory": memory,
        "responses_create_params": {"input": [{"role": "system", "content": system_prompt}]},
    }


def iter_rows(package_dir: Path, *, tier: Optional[str] = None) -> Iterator[dict[str, Any]]:
    tiers = TIERS if tier is None else (tier,)
    if any(t not in TIERS for t in tiers):
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    risks = {row["risk_id"]: row for row in _read_table(package_dir, "risks")}
    for scenario in _scenario_rows(package_dir):
        risk = risks.get(scenario["risk_id"])
        if risk is None:
            raise UpstreamError(f"{scenario['scenario_id']}: risk {scenario['risk_id']} not in taxonomy")
        if int(scenario["planned_turns"]) < 1:
            raise UpstreamError(f"{scenario['scenario_id']}: planned_turns {scenario['planned_turns']}")
        if not str(scenario["first_user_message"]).strip():
            raise UpstreamError(f"{scenario['scenario_id']}: empty first_user_message")
        for current_tier in tiers:
            yield build_row(scenario, risk, tier=current_tier)


def example_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The committed 5-row smoke sample: the first scenario of each risk in EXAMPLE_RISKS."""
    by_key = {(row["risk_id"], row["tier"]): row for row in reversed(rows)}  # first scenario_id wins
    chosen = []
    for risk_id, tier in EXAMPLE_RISKS:
        try:
            chosen.append(by_key[(risk_id, tier)])
        except KeyError:
            raise UpstreamError(f"No row for example risk {risk_id!r} tier {tier!r}") from None
    return chosen


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def prepare(
    *,
    tier: Optional[str] = None,
    upstream_dir: Path = DEFAULT_UPSTREAM_DIR,
    output_fpath: Path = OUTPUT_FPATH,
    server_data_dir: Optional[Path] = DEFAULT_SERVER_DATA_DIR,
    zip_path: Optional[Path] = None,
    download: bool = True,
) -> Path:
    """Fetch and verify the package, export the pack, and write the benchmark JSONL.

    Returns ``output_fpath``, which ``gym eval prepare --benchmark kora`` requires to equal
    the benchmark config's ``jsonl_fpath``. When ``server_data_dir`` is set, the same rows
    are also written there as ``kora.jsonl`` (the resources server's validation split) and
    the five example rows as ``example.jsonl``.
    """
    package_dir = ensure_package(upstream_dir, zip_path=zip_path, download=download)
    write_pack(package_dir, upstream_dir / "pack.json")

    rows = list(iter_rows(package_dir, tier=tier))
    expected = EXPECTED_ROWS if tier is None else EXPECTED_SCENARIOS
    if len(rows) != expected:
        raise UpstreamError(f"Built {len(rows)} rows, expected {expected}")
    if len({row["kora_id"] for row in rows}) != len(rows):
        raise UpstreamError("Duplicate kora_id in rows")

    _write_jsonl(output_fpath, rows)
    logger.info("Wrote %d rows to %s", len(rows), output_fpath)
    if server_data_dir is not None:
        _write_jsonl(server_data_dir / "kora.jsonl", rows)
        if tier is None:
            _write_jsonl(server_data_dir / "example.jsonl", example_rows(rows))
    return output_fpath


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, default=None, help="Build one tier only (default: both).")
    parser.add_argument("--package-zip", type=Path, default=None, help="A local copy of the run package zip.")
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    parser.add_argument("--server-data-dir", type=Path, default=DEFAULT_SERVER_DATA_DIR)
    parser.add_argument("--no-download", action="store_true", help="Fail instead of fetching a missing package.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = prepare(
        tier=args.tier,
        upstream_dir=args.upstream_dir,
        output_fpath=args.output,
        server_data_dir=args.server_data_dir,
        zip_path=args.package_zip,
        download=not args.no_download,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
