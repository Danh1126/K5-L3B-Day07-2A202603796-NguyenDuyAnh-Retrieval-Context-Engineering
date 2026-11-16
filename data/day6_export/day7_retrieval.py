import json
import os
import shutil
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


def load_env_file():
  possible_paths = [
      ".env",
      "../../.env",
      r"D:\code\K5-L3B-DAY06-2A202603796-DataIngestion\starter\.env",
      r"D:\code\K5-L3B-DAY06-2A202603796-DataIngestion\.env",
  ]
  for path in possible_paths:
    if os.path.exists(path):
      print(f"Đã nạp biến môi trường từ: {path}")
      with open(path, "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
      break


load_env_file()

with open("day6_chunks.json", "r", encoding="utf-8") as f:
  chunks = json.load(f)

with open("day6_questions.json", "r", encoding="utf-8") as f:
  questions = json.load(f)

print(f"Loaded {len(chunks)} chunks và {len(questions)} questions.")

openai_client = OpenAI()
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536


def get_embeddings(texts: list[str]) -> list[list[float]]:
  cleaned = [t.replace("\n", " ") for t in texts]
  response = openai_client.embeddings.create(input=cleaned, model=EMBED_MODEL)
  ordered = sorted(response.data, key=lambda d: d.index)
  return [item.embedding for item in ordered]


QDRANT_PATH = "./qdrant_day7_db"
COLLECTION_NAME = "edu_history_C"
qdrant = QdrantClient(path=QDRANT_PATH)

need_build = True
if qdrant.collection_exists(COLLECTION_NAME):
  if qdrant.count(collection_name=COLLECTION_NAME).count > 0:
    need_build = False
  else:
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

if need_build:
  print(f"Đang tạo Qdrant index từ day6_chunks.json bằng {EMBED_MODEL}...")
  qdrant.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
  )

  batch_size = 64
  points = []
  for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i : i + batch_size]
    batch_vectors = get_embeddings([c["text"] for c in batch_chunks])
    for offset, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors)):
      points.append(PointStruct(id=i + offset, vector=vec, payload=chunk))

  qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
  print(f"Đã lưu xong {len(points)} chunks vào '{QDRANT_PATH}'.")

sample_point = qdrant.scroll(collection_name=COLLECTION_NAME, limit=1)[0][0]
payload_keys = sample_point.payload.keys()
CHAPTER_KEY = "chapter" if "chapter" in payload_keys else "metadata.chapter"



def build_qdrant_filter(q_filter: dict | None) -> Filter | None:
  if not q_filter:
    return None
  conditions = []
  for k, v in q_filter.items():
    key = CHAPTER_KEY if k == "chapter" else k
    conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
  return Filter(must=conditions)


def retrieve_chunks(
    query_text: str, q_filter: dict | None = None, top_k: int = 5
):
  query_vector = get_embeddings([query_text])[0]
  search_result = qdrant.query_points(
      collection_name=COLLECTION_NAME,
      query=query_vector,
      query_filter=build_qdrant_filter(q_filter),
      limit=top_k,
  ).points
  return [
      {
          "chunk_id": hit.payload.get("chunk_id"),
          "page": hit.payload.get("page"),
          "text": hit.payload.get("text"),
          "score": hit.score,
      }
      for hit in search_result
  ]



def evaluate_retrieval(questions_list, top_k: int = 5):
  answerable = [
      q for q in questions_list if q.get("answer") != "INSUFFICIENT_EVIDENCE"
  ]
  hits_at_k = 0
  mrr_total = 0.0

  for q in answerable:
    results = retrieve_chunks(q["question"], q.get("filter"), top_k=top_k)
    gold_page = q["gold_page"]

    rank = None
    for idx, res in enumerate(results, start=1):
      if res["page"] == gold_page or (
          q.get("evidence_quote") and q["evidence_quote"][:40] in res["text"]
      ):
        rank = idx
        break

    if rank is not None:
      hits_at_k += 1
      mrr_total += 1.0 / rank

  recall_at_k = hits_at_k / len(answerable)
  mrr = mrr_total / len(answerable)
  return {"Recall@K": round(recall_at_k, 4), "MRR": round(mrr, 4)}


if __name__ == "__main__":
  metrics = evaluate_retrieval(questions, top_k=5)
  print("Kết quả Baseline Retrieval:", metrics)