from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "starter"))
from baseline import pack_context
from real_agent import QdrantDenseRetriever, load_env_file


class BM25Retriever:
  """Phương pháp 2: Lexical BM25 Retriever tương thích giao diện Retriever."""

  def __init__(
      self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75
  ) -> None:
    self.chunks = chunks
    self.k1 = k1
    self.b = b
    self.docs_tokens = [self._tokenize(c["text"]) for c in chunks]
    self.doc_lens = [len(t) for t in self.docs_tokens]
    self.avgdl = (
        sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0
    )
    self.doc_freqs: Counter[str] = Counter()
    for tokens in self.docs_tokens:
      for tok in set(tokens):
        self.doc_freqs[tok] += 1
    self.N = len(chunks)

  @staticmethod
  def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

  def search(
      self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
  ) -> list[dict[str, Any]]:
    if not query or not query.strip():
      return []
    q_tokens = self._tokenize(query)
    scored: list[tuple[float, dict[str, Any]]] = []

    for idx, chunk in enumerate(self.chunks):
      if filters:
        meta = chunk.get("metadata", {})
        chap = chunk.get("chapter") or meta.get("chapter")
        if "chapter" in filters and chap != filters["chapter"]:
          continue

      tf_counter = Counter(self.docs_tokens[idx])
      dl = self.doc_lens[idx]
      score = 0.0
      for tok in q_tokens:
        if tok not in tf_counter:
          continue
        df = self.doc_freqs[tok]
        idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        tf = tf_counter[tok]
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
        score += idf * (tf * (self.k1 + 1.0)) / denom

      if score > 0:
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
      results.append({
          "id": chunk.get("id") or chunk.get("chunk_id"),
          "text": chunk.get("text", ""),
          "source_id": chunk.get("source_id")
          or chunk.get("document_id", ""),
          "source": chunk.get("source", ""),
          "page": chunk.get("page", 0),
          "metadata": chunk.get("metadata", {}),
          "score": float(score),
      })
    return results


class HybridRRFRetriever:
  """Phương pháp 3: Hybrid Retrieval kết hợp Dense + BM25 qua Reciprocal Rank Fusion."""

  def __init__(
      self,
      dense_retriever: QdrantDenseRetriever,
      bm25_retriever: BM25Retriever,
      rrf_k: int = 60,
      dense_weight: float = 0.6,
      bm25_weight: float = 0.4,
  ) -> None:
    self.dense = dense_retriever
    self.bm25 = bm25_retriever
    self.rrf_k = rrf_k
    self.dense_weight = dense_weight
    self.bm25_weight = bm25_weight

  def search(
      self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
  ) -> list[dict[str, Any]]:
    if not query or not query.strip():
      return []
    dense_hits = self.dense.search(query, filters=filters, top_k=top_k * 3)
    bm25_hits = self.bm25.search(query, filters=filters, top_k=top_k * 3)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_hits, start=1):
      cid = item["id"]
      chunk_map[cid] = item
      rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight * (
          1.0 / (self.rrf_k + rank)
      )

    for rank, item in enumerate(bm25_hits, start=1):
      cid = item["id"]
      if cid not in chunk_map:
        chunk_map[cid] = item
      rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight * (
          1.0 / (self.rrf_k + rank)
      )

    ranked_ids = sorted(
        rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True
    )[:top_k]
    results: list[dict[str, Any]] = []
    for cid in ranked_ids:
      row = dict(chunk_map[cid])
      row["score"] = float(rrf_scores[cid])
      results.append(row)
    return results


def evaluate_retriever_on_dev(
    name: str,
    retriever: Any,
    dev_questions: list[dict[str, Any]],
    top_k: int = 3,
    budget: int = 1800,
) -> dict[str, Any]:
  hits_at_1 = 0
  hits_at_k = 0
  context_hits = 0
  mrr_sum = 0.0
  latencies_ms: list[float] = []
  missed_details: list[dict[str, Any]] = []

  for q in dev_questions:
    t0 = time.perf_counter()
    results = retriever.search(
        q["question"], filters=q.get("filter"), top_k=top_k
    )
    latencies_ms.append((time.perf_counter() - t0) * 1000)

    context_str, packed_ids = pack_context(results, budget=budget)
    gold_list = q.get("gold", [])

    def is_match(chunk_row: dict[str, Any]) -> bool:
      return any(
          chunk_row["source_id"] == g["source_id"]
          and chunk_row["page"] == g["page"]
          and g["evidence"] in chunk_row["text"]
          for g in gold_list
      )

    rank = None
    for idx, r in enumerate(results, start=1):
      if is_match(r):
        rank = idx
        break

    if rank is not None:
      hits_at_k += 1
      mrr_sum += 1.0 / rank
      if rank == 1:
        hits_at_1 += 1

    in_packed = any(
        is_match(r) for r in results if r["id"] in set(packed_ids)
    )
    if in_packed:
      context_hits += 1
    else:
      missed_details.append({
          "id": q["id"],
          "question": q["question"],
          "gold_page": gold_list[0]["page"] if gold_list else None,
          "retrieved_rank": rank,
          "top_retrieved_pages": [r["page"] for r in results],
          "packed_count": len(packed_ids),
      })

  n = len(dev_questions)
  return {
      "config": name,
      "n_dev": n,
      "top_k": top_k,
      "budget": budget,
      "Hit@1": round(hits_at_1 / n, 4),
      f"Recall@{top_k}": round(hits_at_k / n, 4),
      "MRR": round(mrr_sum / n, 4),
      "Context_Hit_Rate": round(context_hits / n, 4),
      "Median_Latency_ms": round(statistics.median(latencies_ms), 2),
      "Missed_in_Context": missed_details,
  }


def main() -> None:
  load_env_file()
  with open(
      ROOT_DIR / "starter" / "data" / "converted" / "chunks.json",
      "r",
      encoding="utf-8",
  ) as f:
    chunks = json.load(f)

  with open(
      ROOT_DIR
      / "starter"
      / "data"
      / "converted"
      / "questions_answerable.json",
      "r",
      encoding="utf-8",
  ) as f:
    questions = json.load(f)

  dev_qs = [q for q in questions if q.get("split") == "dev"]
  print(
      f"Đang chạy thử nghiệm 4 cấu hình Retrieval trên {len(dev_qs)} câu DEV...\n"
  )

  oa_client = OpenAI()
  dense_ret = QdrantDenseRetriever(
      qdrant_path=ROOT_DIR / "data" / "day6_export" / "qdrant_day7_db",
      openai_client=oa_client,
  )
  bm25_ret = BM25Retriever(chunks)
  hybrid_default = HybridRRFRetriever(
      dense_ret, bm25_ret, rrf_k=60, dense_weight=0.6, bm25_weight=0.4
  )
  hybrid_tuned = HybridRRFRetriever(
      dense_ret, bm25_ret, rrf_k=30, dense_weight=0.5, bm25_weight=0.5
  )

  try:
    experiments = [
        ("1. Dense (Baseline, k=3, budget=1800)", dense_ret, 3, 1800),
        ("2. Lexical BM25 (k=3, budget=1800)", bm25_ret, 3, 1800),
        (
            "3. Hybrid RRF (0.6/0.4, k=3, budget=1800)",
            hybrid_default,
            3,
            1800,
        ),
        (
            "4. Hybrid RRF Tuned + Budget (0.5/0.5, rrf_k=30, k=5, budget=3000)",
            hybrid_tuned,
            5,
            3000,
        ),
    ]

    all_reports = []
    for label, ret, k, bud in experiments:
      rep = evaluate_retriever_on_dev(label, ret, dev_qs, top_k=k, budget=bud)
      all_reports.append(rep)
      print(
          f"{rep['config']}\n"
          f"   Hit@1={rep['Hit@1']} | Recall@{k}={rep[f'Recall@{k}']} | MRR={rep['MRR']} | "
          f"Context_Hit={rep['Context_Hit_Rate']} | Latency={rep['Median_Latency_ms']}ms\n"
          f"   Missed IDs: {[m['id'] for m in rep['Missed_in_Context']]}\n"
      )

    out_path = (
        ROOT_DIR / "starter" / "results" / "dev_retrieval_experiments.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu báo cáo chi tiết & phân tích lỗi tại: {out_path}")
  finally:
    dense_ret.close()


if __name__ == "__main__":
  main()