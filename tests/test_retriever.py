from types import SimpleNamespace

import pytest

from tools.retriever import (
    ChromaRetrieverService,
    build_retrieval_query,
    reciprocal_rank_fusion,
    tokenize_policy_text,
)


def test_build_retrieval_query_preserves_current_question():
    query = build_retrieval_query(
        "Can I return this product?",
        [
            {"role": "user", "content": "I bought a coffee maker."},
            {"role": "assistant", "content": "When did it arrive?"},
            {"role": "user", "content": "It arrived yesterday."},
        ],
    )

    assert "Can I return this product?" in query
    assert "I bought a coffee maker." in query
    assert "It arrived yesterday." in query
    assert "When did it arrive?" not in query


def test_build_retrieval_query_without_history_returns_question():
    assert build_retrieval_query("  Return policy?  ") == "Return policy?"


def test_build_retrieval_query_uses_only_recent_four_user_messages():
    history = [
        {"role": "user", "content": f"Message {index}"}
        for index in range(6)
    ]

    query = build_retrieval_query("Current question", history)

    assert "Current question" in query
    assert "Message 5" in query
    assert "Message 4" in query
    assert "Message 3" in query
    assert "Message 2" in query
    assert "Message 1" not in query
    assert "Message 0" not in query


def test_build_retrieval_query_ignores_non_user_messages():
    query = build_retrieval_query(
        "Current question",
        [
            {"role": "assistant", "content": "Assistant response"},
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Previous question"},
        ],
    )

    assert "Current question" in query
    assert "Previous question" in query
    assert "Assistant response" not in query
    assert "System instruction" not in query


def test_tokenize_policy_text_removes_stopwords_and_stems():
    tokens = tokenize_policy_text(
        "The products are being shipped and delivered within 30 days."
    )

    assert "the" not in tokens
    assert "are" not in tokens
    assert "product" in tokens
    assert "shipp" in tokens
    assert "deliver" in tokens
    assert "30" in tokens
    assert "day" in tokens


def test_tokenize_policy_text_handles_empty_input():
    assert tokenize_policy_text("") == []
    assert tokenize_policy_text("   ") == []


def test_reciprocal_rank_fusion_combines_vector_and_lexical_results():
    ranked = reciprocal_rank_fusion(
        ["A", "B", "C"],
        ["B", "D", "A"],
        limit=4,
    )

    scores = dict(ranked)

    assert set(scores) == {"A", "B", "C", "D"}
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["B"] > scores["A"]


def test_reciprocal_rank_fusion_respects_result_limit():
    ranked = reciprocal_rank_fusion(
        ["A", "B", "C"],
        ["D", "E", "F"],
        limit=3,
    )

    assert len(ranked) == 3


def test_reciprocal_rank_fusion_deduplicates_shared_identifiers():
    ranked = reciprocal_rank_fusion(
        ["A", "B"],
        ["B", "A"],
    )

    assert len(ranked) == 2
    assert len({identifier for identifier, _ in ranked}) == 2


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error

    def predict(self, pairs):
        if self.error:
            raise self.error
        return self.scores


def _make_retriever(reranker):
    service = ChromaRetrieverService.__new__(ChromaRetrieverService)
    service._reranker = reranker
    return service


def test_reranker_sorts_candidates_by_cross_encoder_score():
    service = _make_retriever(
        FakeReranker(scores=[-2.0, 4.5, 1.2])
    )
    chunks = [
        {
            "id": "A",
            "section": "Returns",
            "subsection": "Shipping",
            "text": "A",
        },
        {
            "id": "B",
            "section": "Returns",
            "subsection": "Eligibility",
            "text": "B",
        },
        {
            "id": "C",
            "section": "Refunds",
            "subsection": "Processing",
            "text": "C",
        },
    ]

    results = service._rerank("return policy", chunks, limit=2)

    assert [item["id"] for item in results] == ["B", "C"]
    assert results[0]["rerank_score"] == pytest.approx(4.5)
    assert results[1]["rerank_score"] == pytest.approx(1.2)


def test_reranker_failure_preserves_original_candidate_order(caplog):
    service = _make_retriever(
        FakeReranker(error=RuntimeError("synthetic failure"))
    )
    chunks = [
        {
            "id": "A",
            "section": "Returns",
            "subsection": "Shipping",
            "text": "A",
        },
        {
            "id": "B",
            "section": "Returns",
            "subsection": "Eligibility",
            "text": "B",
        },
    ]

    results = service._rerank("return policy", chunks, limit=2)

    assert [item["id"] for item in results] == ["A", "B"]
    assert "Local reranker unavailable" in caplog.text


class FakeCollection:
    def get(self, ids, include):
        assert ids == ["LEXICAL-1"]
        assert "embeddings" in include

        return {
            "ids": ["LEXICAL-1"],
            "documents": ["Standard return eligibility and return window."],
            "metadatas": [
                {
                    "section": "RETURN POLICY",
                    "subsection": "Standard Returns",
                    "original_text": "Standard return eligibility and return window.",
                }
            ],
            "embeddings": [[1.2, 0.0]],
        }


def test_hybrid_candidate_generation_preserves_lexical_candidate_above_distance_threshold():
    service = ChromaRetrieverService.__new__(ChromaRetrieverService)
    service.settings = SimpleNamespace(
        retrieval_max_distance=1.35
    )
    service._collection = FakeCollection()
    service._bm25_rank = lambda query, limit: ["LEXICAL-1"]

    results = service._apply_hybrid(
        query="Can I return this product?",
        chunks=[],
        candidate_count=10,
        query_embedding=[0.0, 0.0],
    )

    assert len(results) == 1
    assert results[0]["id"] == "LEXICAL-1"
    assert results[0]["subsection"] == "Standard Returns"
    assert results[0]["distance"] == pytest.approx(1.44)
