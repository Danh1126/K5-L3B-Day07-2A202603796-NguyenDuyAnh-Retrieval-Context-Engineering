import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "starter"))
from agentic import run_agent
from real_agent import OpenAIController, QdrantDenseRetriever, load_env_file


def estimate_cost_usd(embed_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
    embed_cost = (embed_tokens / 1_000_000) * 0.02
    llm_cost = (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.60
    return round(embed_cost + llm_cost, 6)


def main() -> None:
    load_env_file()
    oa_client = OpenAI()

    with open(ROOT_DIR / "data" / "day6_export" / "day6_questions.json", "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    with open(ROOT_DIR / "starter" / "data" / "converted" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_by_id = {c["id"]: c for c in chunks}

    test_questions = [q for q in raw_questions if q.get("split") == "test"]
    if not any(q.get("answer") == "INSUFFICIENT_EVIDENCE" for q in test_questions):
        unans_test = [q for q in raw_questions if q.get("answer") == "INSUFFICIENT_EVIDENCE"][2:]
        for u in unans_test:
            u["split"] = "test"
        test_questions.extend(unans_test)

    print(f"=== CHẠY ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST ({len(test_questions)} câu) ===")
    print("Cấu hình chốt từ DEV: Qdrant Dense (text-embedding-3-small) | top_k=5 | budget=3000 | max_searches=3 | gpt-4o-mini\n")

    retriever = QdrantDenseRetriever(
        qdrant_path=ROOT_DIR / "data" / "day6_export" / "qdrant_day7_db",
        openai_client=oa_client,
    )
    controller = OpenAIController(model="gpt-4o-mini", openai_client=oa_client)

    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    try:
        for q in test_questions:
            qid = q["id"]
            is_answerable = q.get("answer") != "INSUFFICIENT_EVIDENCE"
            cases.append({
                "id": qid,
                "split": "test",
                "answerable": is_answerable,
            })

            t0 = time.perf_counter()
            try:
                res = run_agent(
                    question=q["question"],
                    controller=controller,
                    retriever=retriever,
                    filters=q.get("filter"),
                    max_searches=3,
                    top_k=5,
                    budget=3000,
                )
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                trace_row = {
                    "query_id": qid,
                    "question": q["question"],
                    "expected_answer": q.get("answer"),
                    "status": res["status"],
                    "answer": res["answer"],
                    "citations": res["citations"],
                    "latency_ms": latency_ms,
                    "events": res["events"],
                }
            except Exception as exc:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                trace_row = {
                    "query_id": qid,
                    "question": q["question"],
                    "expected_answer": q.get("answer"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "citations": [],
                    "latency_ms": latency_ms,
                    "events": [],
                }
            traces.append(trace_row)

            if trace_row["status"] != "error":
                cit_support = []
                for cid in trace_row["citations"]:
                    c_obj = chunk_by_id.get(cid)
                    if is_answerable and c_obj:
                        ev = q.get("evidence_quote", "")
                        supported = (c_obj.get("page") == q.get("gold_page")) or (ev[:30] in c_obj.get("text", ""))
                        cit_support.append(bool(supported))
                    else:
                        cit_support.append(False)

                if is_answerable:
                    if trace_row["status"] == "abstain":
                        grade = 0
                    elif any(cit_support) and trace_row["answer"].strip():
                        grade = 2
                    elif trace_row["answer"].strip():
                        grade = 1
                    else:
                        grade = 0
                    reviews.append({
                        "query_id": qid,
                        "citation_support": cit_support,
                        "answer_grade": grade,
                        "reviewer_note": f"Expected: {q.get('answer')} | Agent: {trace_row['answer']}",
                    })
                else:
                    abstain_ok = trace_row["status"] == "abstain"
                    reviews.append({
                        "query_id": qid,
                        "citation_support": cit_support,
                        "abstention_correct": abstain_ok,
                        "reviewer_note": f"Unanswerable check -> status={trace_row['status']}",
                    })

            print(
                f"[{qid}] answerable={is_answerable} | status={trace_row['status']} | "
                f"cits={trace_row['citations']} | {latency_ms}ms\n"
                f"   Q: {q['question']}\n"
                f"   A: {trace_row.get('answer', trace_row.get('error'))}\n"
            )

        out_dir = ROOT_DIR / "starter" / "results" / "agent-test"
        out_dir.mkdir(parents=True, exist_ok=True)

        cases_path = out_dir / "cases.json"
        traces_path = out_dir / "traces.json"
        reviews_path = out_dir / "reviews.json"
        summary_path = out_dir / "summary.json"

        with open(cases_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        with open(traces_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        with open(reviews_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable,
            str(ROOT_DIR / "starter" / "agent_metrics.py"),
            "--cases", str(cases_path),
            "--traces", str(traces_path),
            "--reviews", str(reviews_path),
            "--split", "test",
            "--output", str(summary_path),
        ]
        subprocess.run(cmd, check=True)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        total_cost = estimate_cost_usd(
            embed_tokens=len(test_questions) * 20,
            prompt_tokens=controller.total_prompt_tokens,
            completion_tokens=controller.total_completion_tokens,
        )
        summary["llm_prompt_tokens"] = controller.total_prompt_tokens
        summary["llm_completion_tokens"] = controller.total_completion_tokens
        summary["estimated_total_cost_usd"] = total_cost
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=== TỔNG HỢP METRICS CHÍNH THỨC TRÊN TẬP TEST (agent_metrics.py) ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        retriever.close()


if __name__ == "__main__":
    main()