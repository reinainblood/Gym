# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pickle

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
