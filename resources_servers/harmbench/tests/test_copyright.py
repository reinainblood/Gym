# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pickle
from unittest.mock import patch

import pytest

from resources_servers.harmbench.copyright import CopyrightScorer


def test_upstream_minhash_decision_with_synthetic_reference(tmp_path):
    pytest.importorskip("spacy")
    datasketch = pytest.importorskip("datasketch")
    pytest.importorskip("en_core_web_sm")
    text = "A wholly synthetic phrase about silver lamps and winter gardens."
    nlp = __import__("spacy").load("en_core_web_sm")
    reference = datasketch.MinHash(scheme="legacy")
    for token in nlp(text):
        reference.update(token.text.encode("utf8"))
    raw = pickle.dumps([reference])
    (tmp_path / "synthetic.pkl").write_bytes(raw)

    scorer = CopyrightScorer(tmp_path, verify_checkout=False)
    positive = scorer.score("synthetic", text, ["lyrics", "hash_check"])
    negative = scorer.score("synthetic", "Completely different words are placed here.", ["lyrics", "hash_check"])
    assert positive.label == 1 and positive.max_similarity == 1.0
    assert positive.reference_sha256 == hashlib.sha256(raw).hexdigest()
    assert negative.label == 0 and negative.max_similarity <= 0.6


def test_rejects_invalid_reference_path_and_tags(tmp_path):
    pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    scorer = CopyrightScorer(tmp_path, verify_checkout=False)
    with pytest.raises(ValueError, match="exactly one"):
        scorer.score("anything", "text", ["book", "lyrics", "hash_check"])
    with pytest.raises(ValueError, match="invalid HarmBench behavior ID"):
        scorer.score("../escape", "text", ["book", "hash_check"])


def test_checkout_verification_rejects_wrong_location_and_revision(tmp_path):
    pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    wrong_location = tmp_path / "references"
    wrong_location.mkdir()
    with pytest.raises(ValueError, match="pinned HarmBench checkout"):
        CopyrightScorer(wrong_location)

    reference_dir = tmp_path / "upstream" / "data" / "copyright_classifier_hashes"
    reference_dir.mkdir(parents=True)
    with patch("subprocess.check_output", return_value="wrong-revision\n"):
        with pytest.raises(ValueError, match="does not match pinned HarmBench"):
            CopyrightScorer(reference_dir)


def test_reference_validation_legacy_migration_and_cache(tmp_path):
    pytest.importorskip("spacy")
    datasketch = pytest.importorskip("datasketch")
    pytest.importorskip("en_core_web_sm")
    scorer = CopyrightScorer(tmp_path, verify_checkout=False)

    with pytest.raises(FileNotFoundError):
        scorer._reference("missing")

    (tmp_path / "invalid.pkl").write_bytes(pickle.dumps([]))
    with pytest.raises(ValueError, match="invalid upstream MinHash reference"):
        scorer._reference("invalid")

    legacy = datasketch.MinHash(scheme="legacy")
    del legacy.scheme
    raw = pickle.dumps([legacy])
    (tmp_path / "legacy.pkl").write_bytes(raw)
    first_references, first_hash = scorer._reference("legacy")
    second_references, second_hash = scorer._reference("legacy")
    assert first_references is second_references
    assert first_references[0].scheme == "legacy"
    assert first_hash == second_hash == hashlib.sha256(raw).hexdigest()
