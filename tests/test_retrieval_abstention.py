from types import SimpleNamespace

from tools.retriever import ChromaRetrieverService


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


def make_service(scores, minimum_score):
    service = ChromaRetrieverService.__new__(ChromaRetrieverService)
    service.settings = SimpleNamespace(
        rag_reranker_min_score=minimum_score,
    )
    service._reranker = FakeReranker(scores)
    return service


def candidate(identifier):
    return {
        "id": identifier,
        "section": "Policy",
        "subsection": "Policy section",
        "text": "Policy content.",
        "distance": 1.0,
    }


def test_reranker_abstains_when_top_score_is_below_configured_threshold():
    service = make_service(scores=[-10.8, -11.2], minimum_score=-10.73)

    results = service._rerank(
        "A question with no policy answer",
        [candidate("A"), candidate("B")],
        limit=2,
    )

    assert results == []


def test_reranker_keeps_candidate_at_threshold():
    service = make_service(scores=[-10.73, -11.2], minimum_score=-10.73)

    results = service._rerank(
        "A question with a weak but supported match",
        [candidate("A"), candidate("B")],
        limit=2,
    )

    assert [item["id"] for item in results] == ["A", "B"]
    assert results[0]["rerank_score"] == -10.73


def test_reranker_threshold_is_configurable_per_settings():
    service = make_service(scores=[-10.6, -11.2], minimum_score=-10.5)

    results = service._rerank(
        "A question",
        [candidate("A"), candidate("B")],
        limit=2,
    )

    assert results == []
