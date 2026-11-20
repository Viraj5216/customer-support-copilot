"""Retrieval-only eval: compare retrieval configurations on the SAME policy file.

No LLM and no API key. It runs the local embedding model, Chroma and (optionally)
the cross-encoder, so results are deterministic and the whole thing takes seconds.

    python evals/run_retrieval_eval.py                 # full comparison
    python evals/run_retrieval_eval.py --validate-only # offline dataset check, no models

Each question has `gold_any`: subsection numbers (e.g. "1.3") that legitimately answer it.
A question is a hit at k if any gold subsection appears in the top k results.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "evals" / "data" / "retrieval_30.jsonl"
KB = ROOT / "data" / "company_knowledge_base.txt"
OUT_DIR = ROOT / "evals" / "results"
TOP_N = 10  # rank depth used for MRR; hit@k uses the first k of these

# name, section-aware embeddings, BM25 hybrid, cross-encoder reranker
CONFIGS = [
    ("1. Semantic only (current default)", False, False, False),
    ("2. + section-aware embeddings", True, False, False),
    ("3. + BM25 hybrid", False, True, False),
    ("4. + reranker (no BM25)", False, False, True),
    ("5. BM25 hybrid + reranker", False, True, True),
    ("6. All three", True, True, True),
]


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def subsection_number(chunk: dict) -> str:
    return str(chunk.get("subsection", "")).split()[0]


def first_gold_rank(chunks: list[dict], gold: set[str]) -> int | None:
    for rank, chunk in enumerate(chunks, 1):
        if subsection_number(chunk) in gold:
            return rank
    return None


# ---------------------------------------------------------------- offline validation
def validate(cases: list[dict], kb_path: Path) -> list[str]:
    from tools.indexer import build_policy_chunks

    chunks = build_policy_chunks(kb_path.read_text(encoding="utf-8"), str(kb_path))
    known = {c.subsection.split()[0] for c in chunks}
    errors, seen = [], set()
    for case in cases:
        if case["id"] in seen:
            errors.append(f"{case['id']}: duplicate id")
        seen.add(case["id"])
        if case["answerable"] and not case["gold_any"]:
            errors.append(f"{case['id']}: answerable but no gold sections")
        if not case["answerable"] and case["gold_any"]:
            errors.append(f"{case['id']}: unanswerable but has gold sections")
        for gold in case["gold_any"]:
            if gold not in known:
                errors.append(f"{case['id']}: gold {gold} is not a KB subsection")
    print(f"{len(cases)} questions | {sum(c['answerable'] for c in cases)} answerable | "
          f"{len(chunks)} KB chunks")
    return errors


def lexical_sanity(cases: list[dict], kb_path: Path) -> None:
    """Rough keyword-only baseline (own tokenizer, NOT the app's) to gauge difficulty."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("(rank_bm25 not installed: skipping lexical sanity check)")
        return
    from tools.indexer import build_policy_chunks

    chunks = build_policy_chunks(kb_path.read_text(encoding="utf-8"), str(kb_path))
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    bm25 = BM25Okapi([tok(c.text) for c in chunks])
    answerable = [c for c in cases if c["answerable"]]
    for k in (1, 3, 5):
        hits = 0
        for case in answerable:
            scores = bm25.get_scores(tok(case["question"]))
            top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
            hits += any(chunks[i].subsection.split()[0] in case["gold_any"] for i in top)
        print(f"  keyword-only BM25 sanity: hit@{k} = {hits}/{len(answerable)}")


# ---------------------------------------------------------------------- live eval
def build_service(section_aware: bool, kb_path: Path, chroma_dir: Path):
    from config.settings import Settings
    from tools.indexer import index_policy_file
    from tools.retriever import ChromaRetrieverService

    settings = Settings(
        chroma_db_rel_path=chroma_dir,
        policies_file_rel_path=kb_path,
        rag_section_aware_embeddings=section_aware,
        rag_hybrid_enabled=False,
        rag_reranker_enabled=False,
    )
    service = ChromaRetrieverService(settings)
    index_policy_file(service, kb_path, section_aware_embeddings=section_aware)
    return service


def run_config(service, cases: list[dict], hybrid: bool, rerank: bool) -> list[dict]:
    service.settings.rag_hybrid_enabled = hybrid
    service.settings.rag_reranker_enabled = rerank
    rows = []
    for case in cases:
        started = time.perf_counter()
        chunks = service.retrieve(case["question"], n_results=TOP_N)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rank = first_gold_rank(chunks, set(case["gold_any"])) if case["answerable"] else None
        # The retriever falls back silently when a stage fails (for example the reranker
        # cannot load). Stages tag their output, so check the tags instead of trusting flags.
        missing = []
        if chunks and hybrid and not all("hybrid_score" in c for c in chunks):
            missing.append("hybrid")
        if chunks and rerank and not all("rerank_score" in c for c in chunks):
            missing.append("rerank")
        rows.append({
            "id": case["id"], "type": case["type"], "answerable": case["answerable"],
            "rank": rank, "returned": len(chunks), "ms": elapsed_ms,
            "top3": [c["subsection"] for c in chunks[:3]],
            "stage_missing": missing,
        })
    for stage, enabled in (("hybrid", hybrid), ("rerank", rerank)):
        if not enabled:
            continue
        with_results = [r for r in rows if r["returned"]]
        if with_results and all(stage in r["stage_missing"] for r in with_results):
            raise RuntimeError(
                f"The {stage} stage never ran (no result carried its score), so this "
                f"configuration would silently equal a simpler one. "
                + ("The reranker loads with local_files_only=True; make sure the model is cached "
                   "or let this runner preload it." if stage == "rerank" else "")
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    n = len(ans)
    hit = lambda k: sum(1 for r in ans if r["rank"] is not None and r["rank"] <= k)
    by_type = defaultdict(lambda: [0, 0])
    for r in ans:
        by_type[r["type"]][1] += 1
        by_type[r["type"]][0] += r["rank"] is not None and r["rank"] <= 3
    return {
        "n_answerable": n,
        "hit1": hit(1), "hit3": hit(3), "hit5": hit(5),
        "mrr": sum(1 / r["rank"] for r in ans if r["rank"]) / n,
        "false_abstain": sum(1 for r in ans if r["returned"] == 0),
        "n_unanswerable": len(unans),
        "abstained": sum(1 for r in unans if r["returned"] == 0),
        "median_ms": statistics.median(r["ms"] for r in rows),
        "stage_fallbacks": sum(1 for r in rows if r["stage_missing"]),
        "hit3_by_type": {k: v for k, v in by_type.items()},
    }


def render_markdown(results: dict, meta: dict) -> str:
    base = next(iter(results.values()))["summary"]
    lines = [
        "# Retrieval evaluation (same policy file, different retrieval settings)", "",
        f"- KB: `{meta['kb']}` ({meta['chunks']} chunks), embedding model `{meta['embedder']}`, "
        f"reranker `{meta['reranker']}`",
        f"- Questions: {meta['answerable']} answerable + {meta['unanswerable']} unanswerable. "
        "A hit means a gold subsection is in the top k results.",
        "- No LLM calls. Deterministic apart from model floating-point noise.", "",
        "## Results", "",
        "| Configuration | hit@1 | hit@3 | hit@5 | MRR | median ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, data in results.items():
        s, n = data["summary"], data["summary"]["n_answerable"]
        lines.append(f"| {name} | {s['hit1']}/{n} | {s['hit3']}/{n} | {s['hit5']}/{n} | "
                     f"{s['mrr']:.3f} | {s['median_ms']:.0f} |")
    fell = {n: d["summary"]["stage_fallbacks"] for n, d in results.items() if d["summary"]["stage_fallbacks"]}
    if fell:
        lines += ["", "Questions where an enabled stage silently fell back to the plain ranking: "
                  + ", ".join(f"{n} ({c})" for n, c in fell.items()) + "."]
    lines += ["", "## hit@3 by question type", "",
              "| Configuration | " + " | ".join(sorted(base["hit3_by_type"])) + " |",
              "|---|" + "---:|" * len(base["hit3_by_type"])]
    for name, data in results.items():
        t = data["summary"]["hit3_by_type"]
        lines.append(f"| {name} | " + " | ".join(f"{t[k][0]}/{t[k][1]}" for k in sorted(t)) + " |")
    lines += ["", "## Abstention (questions the KB cannot answer)", "",
              "| Configuration | correctly returned nothing | answerable questions that returned nothing |",
              "|---|---:|---:|"]
    for name, data in results.items():
        s = data["summary"]
        lines.append(f"| {name} | {s['abstained']}/{s['n_unanswerable']} | "
                     f"{s['false_abstain']}/{s['n_answerable']} |")
    lines += ["", "Note: only vector-only retrieval applies the distance cutoff. Hybrid and reranker "
              "modes deliberately skip it, so they always return chunks and cannot abstain at the "
              "retrieval layer.", "", "## Misses (answerable questions with no gold section in the top 5)", ""]
    for name, data in results.items():
        misses = [r for r in data["rows"] if r["answerable"] and (r["rank"] is None or r["rank"] > 5)]
        text = ", ".join(f"{r['id']} (top: {', '.join(r['top3'][:2]) or 'nothing'})" for r in misses)
        lines.append(f"- **{name}**: {text or 'none'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--kb", type=Path, default=KB)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    errors = validate(cases, args.kb)
    if errors:
        print("DATASET PROBLEMS:\n" + "\n".join(f" - {e}" for e in errors))
        return 1
    if args.validate_only:
        lexical_sanity(cases, args.kb)
        print("OK: dataset is consistent with the KB.")
        return 0

    from config.settings import Settings
    from tools.indexer import build_policy_chunks

    tmp = Path(tempfile.mkdtemp(prefix="retrieval_eval_"))
    try:
        services = {}
        results = {}
        reranker = None
        for name, section_aware, hybrid, rerank in CONFIGS:
            if section_aware not in services:
                print(f"indexing ({'section-aware' if section_aware else 'content-only'}) ...")
                services[section_aware] = build_service(section_aware, args.kb, tmp / f"chroma_{int(section_aware)}")
            if rerank and reranker is None:
                # The app loads the cross-encoder with local_files_only=True, which never
                # downloads it. Load it here (downloading if needed) and hand it over, so the
                # reranker configurations really rerank.
                from sentence_transformers import CrossEncoder
                from config.settings import Settings as _S
                print("loading cross-encoder ...")
                reranker = CrossEncoder(_S().rag_reranker_model)
            if rerank:
                services[section_aware]._reranker = reranker
            print(f"running {name} ...")
            rows = run_config(services[section_aware], cases, hybrid, rerank)
            results[name] = {"rows": rows, "summary": summarize(rows)}

        defaults = Settings()
        meta = {
            "kb": str(args.kb.relative_to(ROOT)) if args.kb.is_relative_to(ROOT) else str(args.kb),
            "chunks": len(build_policy_chunks(args.kb.read_text(encoding="utf-8"), "kb")),
            "embedder": defaults.embedding_model_name, "reranker": defaults.rag_reranker_model,
            "answerable": sum(c["answerable"] for c in cases),
            "unanswerable": sum(not c["answerable"] for c in cases),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "retrieval_eval.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        report = render_markdown(results, meta)
        (OUT_DIR / "retrieval_eval.md").write_text(report)
        print("\n" + report)
        print(f"saved: {OUT_DIR / 'retrieval_eval.md'} and retrieval_eval.json")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())