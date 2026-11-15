
"""Hybrid retrieval and reranking for policy documents."""

from functools import lru_cache
import logging
import math
import re
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings
from tools.indexer import policy_index_metadata

logger = logging.getLogger(__name__)


def build_retrieval_query(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Include recent customer context without losing the current question."""
    message = (question or "").strip()

    if not history:
        return message

    previous = [
        str(item.get("content", "")).strip()
        for item in history
        if isinstance(item, dict)
        and item.get("role") == "user"
        and str(item.get("content", "")).strip()
    ]

    if not previous:
        return message

    recent = "\n".join(
        f"- {turn}" for turn in reversed(previous[-4:])
    )

    return (
        f"Current customer message (primary):\n{message}\n"
        f"Recent customer messages (newest first):\n{recent}"
    )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for",
    "from", "have", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "the", "to", "we", "what", "when", "where", "who",
    "with", "within", "you",
}


def tokenize_policy_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    result = []

    for token in tokens:
        if token in _STOPWORDS:
            continue

        if token.isdigit():
            result.append(token)
        elif len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            result.append(
                stem[:-1]
                if len(stem) > 2 and stem[-1] == stem[-2]
                else stem
            )
        elif len(token) > 4 and token.endswith("ed"):
            result.append(token[:-2])
        elif len(token) >= 4 and token.endswith("s"):
            result.append(token[:-1])
        else:
            result.append(token)

    return result


def reciprocal_rank_fusion(
    vector_ids: list[str],
    lexical_ids: list[str],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[str, float]]:
    scores = {}

    for ranked in (vector_ids, lexical_ids):
        for rank, identifier in enumerate(ranked, 1):
            scores[identifier] = (
                scores.get(identifier, 0.0) + 1 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]


class ChromaRetrieverService:
    """Retrieve relevant policy chunks from ChromaDB."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._embedder = SentenceTransformer(
            self.settings.embedding_model_name
        )
        self._client = chromadb.PersistentClient(
            path=str(self.settings.chroma_db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection_name
        )

        self._bm25 = None
        self._bm25_rows = []
        self._bm25_metadata = None
        self._reranker = None
        self.index_stale = self._check_staleness()

    def _check_staleness(self) -> bool:
        try:
            expected = policy_index_metadata(
                self.settings.policies_file_path.read_bytes(),
                section_aware_embeddings=(
                    self.settings.rag_section_aware_embeddings
                ),
            )
            metadata = self._collection.metadata or {}

            stale = any(
                metadata.get(key) != value
                for key, value in expected.items()
            )

            if stale:
                logger.warning(
                    "Policy index is stale; run the offline indexer."
                )

            return stale

        except OSError:
            logger.warning("Policy index freshness could not be checked.")
            return True

    @property
    def embedder(self) -> SentenceTransformer:
        return self._embedder

    @property
    def collection(self) -> Any:
        return self._collection

    def retrieve(
        self,
        question: str,
        n_results: int = 5,
        history: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant policy chunks using vector and optional hybrid search."""
        if (
            not isinstance(n_results, int)
            or isinstance(n_results, bool)
            or n_results < 1
        ):
            raise ValueError("n_results must be a positive integer.")

        self.index_stale = self._check_staleness()

        if self.index_stale:
            raise RuntimeError(
                "Policy index is stale or unavailable. Re-index policies."
            )

        query = build_retrieval_query(question, history)
        encoded = self.embedder.encode(query)
        embedding = (
            encoded.tolist()
            if hasattr(encoded, "tolist")
            else list(encoded)
        )

        use_hybrid = self.settings.rag_hybrid_enabled
        use_reranker = self.settings.rag_reranker_enabled

        candidate_count = (
            max(n_results, 10)
            if use_hybrid or use_reranker
            else n_results
        )

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=candidate_count,
        )

        documents = (results.get("documents") or [[]])[0]

        if not documents:
            return []

        ids = (results.get("ids") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        threshold = self.settings.retrieval_max_distance
        chunks = []

        for index, document in enumerate(documents):
            if not document:
                continue

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            if threshold is not None:
                if distance is None:
                    continue

                try:
                    distance = float(distance)
                except (TypeError, ValueError):
                    continue

                if not math.isfinite(distance) or distance > threshold:
                    continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            ) or {}

            chunks.append({
                "id": ids[index] if index < len(ids) else str(index),
                "text": metadata.get("original_text") or document,
                "section": metadata.get("section", "Unknown"),
                "subsection": metadata.get("subsection", "Unknown"),
                "distance": (
                    float(distance) if distance is not None else None
                ),
            })

        if use_hybrid and chunks:
            chunks = self._apply_hybrid(
                query, chunks, candidate_count
            )

        if use_reranker and chunks:
            chunks = self._rerank(query, chunks, n_results)

        return chunks[:n_results]

    def _apply_hybrid(
        self,
        query: str,
        chunks: list[dict],
        candidate_count: int,
    ) -> list[dict]:
        lexical_ids = self._bm25_rank(query, candidate_count)

        if not lexical_ids:
            return chunks

        vector_ids = [item["id"] for item in chunks]

        ranked = reciprocal_rank_fusion(
            vector_ids,
            lexical_ids,
            limit=candidate_count,
        )

        by_id = {item["id"]: item for item in chunks}

        # Do not add lexical-only chunks when a vector-distance
        # threshold is configured. Their relevance has not been
        # verified against that threshold.
        if self.settings.retrieval_max_distance is None:
            missing = [
                identifier
                for identifier, _ in ranked
                if identifier not in by_id
            ]

            if missing:
                extra = self.collection.get(
                    ids=missing,
                    include=["documents", "metadatas"],
                )

                extra_ids = extra.get("ids", [])
                documents = extra.get("documents", [])
                metadatas = extra.get("metadatas", [])

                for index, identifier in enumerate(extra_ids):
                    if index >= len(documents) or not documents[index]:
                        continue

                    metadata = (
                        metadatas[index]
                        if index < len(metadatas)
                        else {}
                    ) or {}

                    by_id[identifier] = {
                        "id": identifier,
                        "text": (
                            metadata.get("original_text")
                            or documents[index]
                        ),
                        "section": metadata.get("section", "Unknown"),
                        "subsection": metadata.get(
                            "subsection", "Unknown"
                        ),
                        "distance": None,
                    }

        return [
            {**by_id[identifier], "hybrid_score": score}
            for identifier, score in ranked
            if identifier in by_id
        ]

    def _bm25_rank(
        self,
        query: str,
        limit: int,
    ) -> list[str] | None:
        metadata = self.collection.metadata or {}

        signature = (
            metadata.get("kb_sha256"),
            metadata.get("chunking_config"),
        )

        if signature != self._bm25_metadata:
            self._bm25 = None
            self._bm25_rows = []
            self._bm25_metadata = signature

        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi

                data = self.collection.get(
                    include=["documents", "metadatas"]
                )

                documents = data.get("documents") or []
                metadatas = data.get("metadatas") or []

                self._bm25_rows = []
                tokenized = []

                for index, identifier in enumerate(
                    data.get("ids", [])
                ):
                    if index >= len(documents) or not documents[index]:
                        continue

                    meta = (
                        metadatas[index]
                        if index < len(metadatas)
                        else {}
                    ) or {}

                    text = (
                        f"{meta.get('section', '')} > "
                        f"{meta.get('subsection', '')} "
                        f"{documents[index]}"
                    )

                    tokens = tokenize_policy_text(text)

                    if not tokens:
                        continue

                    self._bm25_rows.append(identifier)
                    tokenized.append(tokens)

                self._bm25 = (
                    BM25Okapi(tokenized) if tokenized else False
                )

            except Exception as exc:
                logger.warning(
                    "Hybrid lexical retrieval unavailable (%s)",
                    type(exc).__name__,
                )
                return None

        if self._bm25 is False:
            return None

        scores = self._bm25.get_scores(
            tokenize_policy_text(query)
        )

        indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        return [
            self._bm25_rows[index]
            for index in indexes
            if scores[index] > 0
        ]

    def _rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.settings.rag_reranker_model,
                    local_files_only=True,
                )

            pairs = [
                (
                    query,
                    f"{item['section']} > "
                    f"{item['subsection']}\n{item['text']}",
                )
                for item in chunks
            ]

            scores = self._reranker.predict(pairs)

            if len(scores) != len(chunks):
                raise ValueError(
                    "Reranker returned an invalid score count."
                )

            ranked = [
                {**item, "rerank_score": float(score)}
                for item, score in zip(chunks, scores)
            ]

            return sorted(
                ranked,
                key=lambda item: item["rerank_score"],
                reverse=True,
            )[:limit]

        except Exception as exc:
            logger.warning(
                "Local reranker unavailable (%s)",
                type(exc).__name__,
            )
            return chunks[:limit]

    def search_policy_text(
        self,
        question: str,
        n_results: int = 5,
    ) -> str:
        """Format retrieved policy chunks."""
        chunks = self.retrieve(question, n_results=n_results)

        if not chunks:
            return "No relevant policy documents found."

        output = []

        for index, chunk in enumerate(chunks, 1):
            distance = chunk.get("distance")

            distance_text = (
                f"{distance:.4f}"
                if isinstance(distance, (int, float))
                else "unavailable"
            )

            output.append(
                f"Result {index}\n"
                f"Section: {chunk['section']}\n"
                f"Subsection: {chunk['subsection']}\n"
                f"Distance: {distance_text}\n"
                f"Policy:\n{chunk['text']}"
            )

        return "\n\n--------------------\n\n".join(output)


@lru_cache()
def get_retriever_service() -> ChromaRetrieverService:
    """Return the cached application retriever."""
    return ChromaRetrieverService()
