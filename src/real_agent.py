import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# Cho phép import run_agent từ thư mục starter/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "starter"))
from agentic import run_agent


def load_env_file() -> None:
    possible_paths = [
        ROOT_DIR / ".env",
        Path(r"D:\code\K5-L3B-DAY06-2A202603796-DataIngestion\starter\.env"),
        Path(r"D:\code\K5-L3B-DAY06-2A202603796-DataIngestion\.env"),
    ]
    for path in possible_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


class QdrantDenseRetriever:
    """Real Dense Retriever nối tới Qdrant Local và OpenAI Embeddings."""

    def __init__(
        self,
        qdrant_path: str | Path,
        collection_name: str = "edu_history_C",
        embed_model: str = "text-embedding-3-small",
        openai_client: OpenAI | None = None,
    ) -> None:
        self.client = QdrantClient(path=str(qdrant_path))
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.openai = openai_client or OpenAI()

        sample = self.client.scroll(collection_name=self.collection_name, limit=1)[0][0]
        self.chapter_key = "chapter" if "chapter" in sample.payload else "metadata.chapter"

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        conditions = []
        for k, v in filters.items():
            key = self.chapter_key if k == "chapter" else k
            conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
        return Filter(must=conditions)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        resp = self.openai.embeddings.create(
            input=[query.replace("\n", " ")], model=self.embed_model
        )
        q_vec = resp.data[0].embedding
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=self._build_filter(filters),
            limit=top_k,
        ).points

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in hits:
            payload = hit.payload or {}
            cid = payload.get("id") or payload.get("chunk_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            results.append({
                "id": cid,
                "text": payload.get("text", ""),
                "source_id": payload.get("source_id") or payload.get("document_id", ""),
                "source": payload.get("source", ""),
                "page": payload.get("page", 0),
                "metadata": payload.get("metadata", {}),
                "score": float(hit.score),
            })
        return results[:top_k]

    def close(self) -> None:
        self.client.close()


class OpenAIController:
    """Real LLM Controller quyết định search hoặc trả lời kèm citation chuẩn."""

    SYSTEM_PROMPT = """You are a strict historical QA retrieval agent.
You must answer questions ONLY using the retrieved context chunks provided in the history.
Never use outside knowledge or guess.

Output MUST be a valid JSON object matching ONE of these two formats:
1. To search for evidence (if context is empty or does not contain enough facts to answer):
   {"action": "search", "query": "<focused search query>"}
2. To give the final response:
   {"action": "answer", "answer": "<concise answer citing exact chunk IDs like [gutenberg-27963::C::3::0]>", "citations": ["gutenberg-27963::C::3::0"]}

Strict rules:
- Every factual claim in "answer" MUST cite its source chunk ID in square brackets, e.g., [gutenberg-27963::C::3::0], AND list that exact ID in "citations".
- Only cite IDs present in "context_ids".
- If you have already searched and the retrieved context does NOT contain the answer, DO NOT guess. Return:
  {"action": "answer", "answer": "INSUFFICIENT_EVIDENCE", "citations": []}
"""

    def __init__(self, model: str = "gpt-4o-mini", openai_client: OpenAI | None = None) -> None:
        self.model = model
        self.openai = openai_client or OpenAI()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            user_msg = (
                f"Question: {question}\n\n"
                "Retrieved Context: None yet. Call 'search' first to find relevant passages."
            )
        else:
            latest = history[-1]
            user_msg = (
                f"Question: {question}\n\n"
                f"Previous Search Queries: {json.dumps(latest.get('queries', []))}\n"
                f"Available Context IDs: {json.dumps(latest.get('context_ids', []))}\n"
                f"Retrieved Context:\n{latest.get('context', '')}\n\n"
                "Decide whether to search again with a refined query or provide the final answer/abstention."
            )

        resp = self.openai.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        if resp.usage:
            self.total_prompt_tokens += resp.usage.prompt_tokens
            self.total_completion_tokens += resp.usage.completion_tokens

        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        action = data.get("action")
        if action == "search":
            return {"action": "search", "query": str(data.get("query", question))}
        if action == "answer":
            cits = data.get("citations") or []
            return {
                "action": "answer",
                "answer": str(data.get("answer", "")),
                "citations": [str(c) for c in cits],
            }
        raise ValueError(f"Model trả về action không hợp lệ: {data}")


def run_dev_demo() -> None:
    load_env_file()
    oa_client = OpenAI()
    retriever = QdrantDenseRetriever(
        qdrant_path=ROOT_DIR / "data" / "day6_export" / "qdrant_day7_db",
        openai_client=oa_client,
    )
    controller = OpenAIController(model="gpt-4o-mini", openai_client=oa_client)

    with open(ROOT_DIR / "data" / "day6_export" / "day6_questions.json", "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    dev_ans = [q for q in all_questions if q.get("split") == "dev" and q.get("answer") != "INSUFFICIENT_EVIDENCE"][:3]
    unans_qs = [q for q in all_questions if q.get("answer") == "INSUFFICIENT_EVIDENCE"][:2]
    sample_batch = dev_ans + unans_qs

    traces = []
    try:
        for q in sample_batch:
            t0 = time.perf_counter()
            try:
                res = run_agent(
                    question=q["question"],
                    controller=controller,
                    retriever=retriever,
                    filters=q.get("filter"),
                    max_searches=3,
                    top_k=3,
                    budget=1800, 
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                trace_entry = {
                    "query_id": q["id"],
                    "question": q["question"],
                    "expected": q.get("answer"),
                    "status": res["status"],
                    "answer": res["answer"],
                    "citations": res["citations"],
                    "latency_ms": elapsed_ms,
                    "events": res["events"],
                }
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                trace_entry = {
                    "query_id": q["id"],
                    "question": q["question"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "citations": [],
                    "latency_ms": elapsed_ms,
                    "events": [],
                }
            traces.append(trace_entry)
            print(f"[{trace_entry['query_id']}] status={trace_entry['status']} | citations={trace_entry['citations']} | answer={trace_entry['answer'][:90]}")

        out_dir = ROOT_DIR / "starter" / "results" / "agent-dev-demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "traces.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu full traces (gồm tool call, tham số, kết quả, context cuối) tại: {out_path}")
    finally:
        retriever.close()


if __name__ == "__main__":
    run_dev_demo()