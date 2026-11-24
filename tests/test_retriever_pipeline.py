
from types import SimpleNamespace

from tools.retriever import ChromaRetrieverService


class FakeEmbedder:
    def encode(self, query):
        return [0.0, 0.0]


class FakeCollection:
    def __init__(self, results, count=71):
        self.results = results
        self.item_count = count
        self.requested_count = None

    def count(self):
        return self.item_count

    def query(self, query_embeddings, n_results):
        self.requested_count = n_results
        return self.results


def make_service(*, hybrid, reranker, results, count=71):
    service = ChromaRetrieverService.__new__(ChromaRetrieverService)
    service.settings = SimpleNamespace(
        rag_hybrid_enabled=hybrid,
        rag_reranker_enabled=reranker,
        retrieval_max_distance=1.35,
    )
    service._embedder = FakeEmbedder()
    service._collection = FakeCollection(results, count=count)
    service._check_staleness = lambda: False
    service._apply_hybrid = lambda query, chunks, candidate_count, embedding: [
        {
            "id": "LEXICAL-1",
            "text": "Standard return eligibility and return window.",
            "section": "RETURN POLICY",
            "subsection": "Standard Returns",
            "distance": 1.44,
            "hybrid_score": 0.031,
        }
    ]
    service._rerank = lambda query, chunks, limit: chunks[:limit]
    return service


def test_hybrid_retrieval_keeps_relevant_candidate_above_vector_threshold():
    service = make_service(
        hybrid=True,
        reranker=False,
        results={
            "ids": [["VECTOR-1"]],
            "documents": [["Some vector result."]],
            "metadatas": [[{
                "section": "OTHER",
                "subsection": "Other",
            }]],
            "distances": [[0.8]],
        },
    )

    results = service.retrieve(
        "What is the standard return eligibility period?",
        n_results=5,
    )

    assert len(results) == 1
    assert results[0]["id"] == "LEXICAL-1"
    assert results[0]["distance"] == 1.44


def test_hybrid_retrieval_uses_full_collection_candidate_pool():
    service = make_service(
        hybrid=True,
        reranker=False,
        results={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        },
        count=71,
    )

    service.retrieve("What is the warranty period?", n_results=5)

    assert service.collection.requested_count == 71


def test_vector_only_retrieval_keeps_existing_distance_filter():
    service = make_service(
        hybrid=False,
        reranker=False,
        results={
            "ids": [["NEAR", "FAR"]],
            "documents": [["Near result.", "Far result."]],
            "metadatas": [[
                {"section": "A", "subsection": "Near"},
                {"section": "B", "subsection": "Far"},
            ]],
            "distances": [[0.8, 1.44]],
        },
    )

    results = service.retrieve("Return policy", n_results=5)

    assert [item["id"] for item in results] == ["NEAR"]
