"""Deterministic, idempotent policy chunking and Chroma indexing helpers."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNKING_CONFIG = "major-subsection-v1"
_MAJOR_RE = re.compile(r"^\s*(\d+)\.\s+([A-Z][A-Z0-9 &/()\-]*POLICY|[A-Z][A-Z0-9 &/()\-]+SUPPORT POLICY)\s*$")
_SUBSECTION_RE = re.compile(r"^\s*(\d+\.\d+)\s+(.+?)\s*$")


@dataclass(frozen=True)
class PolicyChunk:
    section: str
    subsection: str
    text: str
    source: str
    content_hash: str
    chunk_id: str

    @property
    def embedding_text(self) -> str:
        return f"{self.section} > {self.subsection}\n{self.text}"


def build_policy_chunks(text: str, source: str) -> list[PolicyChunk]:
    section = "Uncategorized"
    subsection = "General"
    lines: list[str] = []
    chunks: list[PolicyChunk] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if (
            not body
            or section == "Uncategorized"
            or re.fullmatch(r"[=*_\-\s]+", body)
        ):
            lines.clear()
            return
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        identity = f"{source}\0{section}\0{subsection}\0{content_hash}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        chunks.append(PolicyChunk(section, subsection, body, source, content_hash, chunk_id))
        lines.clear()

    for line in text.splitlines():
        major = _MAJOR_RE.match(line)
        sub = _SUBSECTION_RE.match(line)
        if major:
            flush()
            section = f"{major.group(1)}. {major.group(2).strip()}"
            subsection = "Overview"
        elif sub:
            flush()
            subsection = f"{sub.group(1)} {sub.group(2).strip()}"
        else:
            lines.append(line)
    flush()
    return chunks


def policy_index_metadata(
    kb_bytes: bytes, *, section_aware_embeddings: bool = False
) -> dict[str, str]:
    embedding_mode = "section-aware" if section_aware_embeddings else "content-only"
    return {
        "kb_sha256": hashlib.sha256(kb_bytes).hexdigest(),
        "chunking_config": f"{CHUNKING_CONFIG}:{embedding_mode}",
    }


def index_policy_file(
    service: Any,
    path: Path | None = None,
    *,
    section_aware_embeddings: bool | None = None,
) -> dict[str, int | str]:
    """Upsert stable chunks and remove stale IDs; not called at runtime."""
    policy_path = path or service.settings.policies_file_path
    kb_bytes = policy_path.read_bytes()
    sectioned = (
        service.settings.rag_section_aware_embeddings
        if section_aware_embeddings is None
        else section_aware_embeddings
    )
    metadata = policy_index_metadata(kb_bytes, section_aware_embeddings=sectioned)
    source = str(policy_path)
    chunks = build_policy_chunks(kb_bytes.decode("utf-8"), source)
    current = service.collection.get(include=["metadatas"])
    current_ids = set(current.get("ids", []))
    desired_ids = {chunk.chunk_id for chunk in chunks}
    stale_ids = current_ids - desired_ids
    embedded_texts = [
        chunk.embedding_text if sectioned else chunk.text for chunk in chunks
    ]
    vectors = service.embedder.encode(embedded_texts).tolist()
    if chunks:
        service.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=embedded_texts,
            embeddings=vectors,
            metadatas=[
                {
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "source": chunk.source,
                    "original_text": chunk.text,
                    "content_hash": chunk.content_hash,
                    "chunking_config": CHUNKING_CONFIG,
                }
                for chunk in chunks
            ],
        )
    if stale_ids:
        service.collection.delete(ids=sorted(stale_ids))
    service.collection.modify(metadata={**(service.collection.metadata or {}), **metadata})
    return {**metadata, "upserted": len(chunks), "removed": len(stale_ids)}


def main() -> None:
    from tools.retriever import get_retriever_service

    service = get_retriever_service()
    result = index_policy_file(service)
    print(f"Indexed {result['upserted']} policy chunks; removed {result['removed']} stale chunks.")


if __name__ == "__main__":
    main()
