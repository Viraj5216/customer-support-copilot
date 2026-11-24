"""Fresh-install checks that keep databases and index artifacts in tmp_path."""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from config import REPO_ROOT
from scripts.migrate_users import migrate_and_seed
from scripts.seed_demo_users import seed_demo_users
from tools.indexer import index_policy_file


def row_counts(database):
    with sqlite3.connect(database) as connection:
        return (
            connection.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        )


def test_seed_creates_four_users_and_thirteen_orders_idempotently(tmp_path):
    database = tmp_path / "fresh" / "orders.db"
    credentials = tmp_path / "fresh" / "demo_credentials.txt"
    now = datetime(2026, 10, 15, 12, tzinfo=ZoneInfo("Asia/Kolkata"))

    seed_demo_users(database, credentials, now=now)
    assert row_counts(database) == (4, 13)
    assert credentials.is_file()

    seed_demo_users(database, credentials, now=now)
    assert row_counts(database) == (4, 13)


def test_migration_entry_point_reuses_seed_logic(tmp_path):
    database = tmp_path / "migrated" / "orders.db"
    credentials = tmp_path / "migrated" / "credentials.txt"
    now = datetime(2026, 10, 15, 12, tzinfo=ZoneInfo("Asia/Kolkata"))
    seed_demo_users(database, credentials, now=now)

    migrate_and_seed(database, credentials)

    assert row_counts(database) == (4, 13)


class Matrix:
    def tolist(self):
        return [[0.0] for _ in range(71)]


class FakeEmbedder:
    def encode(self, texts):
        assert len(texts) == 71
        return Matrix()


class FakeCollection:
    def __init__(self):
        self.metadata = {}
        self.ids = []
        self.upsert_count = 0

    def get(self, *, include):
        assert include == ["metadatas"]
        return {"ids": self.ids}

    def upsert(self, *, ids, documents, embeddings, metadatas):
        self.ids = list(ids)
        self.upsert_count = len(ids)
        assert len(documents) == len(embeddings) == len(metadatas) == 71

    def modify(self, *, metadata):
        self.metadata = metadata


def test_indexer_chunks_and_upserts_seventy_one_policy_sections(tmp_path):
    source = REPO_ROOT / "data" / "company_knowledge_base.txt"
    temporary_policy = tmp_path / "company_knowledge_base.txt"
    shutil.copyfile(source, temporary_policy)
    collection = FakeCollection()
    service = SimpleNamespace(
        settings=SimpleNamespace(rag_section_aware_embeddings=False),
        collection=collection,
        embedder=FakeEmbedder(),
    )

    result = index_policy_file(service, temporary_policy)

    assert result["upserted"] == 71
    assert result["removed"] == 0
    assert collection.upsert_count == 71
    assert len(collection.ids) == 71
