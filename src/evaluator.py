import json
import pickle
import chromadb
import pandas as pd
import numpy as np
import random
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.run_config import RunConfig
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# CrossEncoder 전역 캐싱
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# --- Vector RAG ---
def vector_ask(query: str, collection, embeddings, top_k: int = 5) -> dict:
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results["documents"][0]
    context = "\n\n---\n\n".join(chunks)

    llm = OllamaLLM(model="qwen2.5:14b", timeout=180, num_predict=512)
    prompt = PromptTemplate(
        template="""Answer the question based only on the context below.
If the answer is not in the context, say 'I cannot find the information.'

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]""",
        input_variables=["context", "question"]
    )
    answer = (prompt | llm).invoke({"context": context, "question": query})
    return {"answer": answer, "contexts": chunks}


# --- Hybrid RAG ---
def hybrid_ask_eval(query: str, collection, embeddings, bm25_data, top_k: int = 5) -> dict:
    # Vector Search
    query_embedding = embeddings.embed_query(query)
    v_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas"]
    )
    vector_results = [
        {"chunk": doc, "metadata": meta}
        for doc, meta in zip(v_results["documents"][0], v_results["metadatas"][0])
    ]

    # BM25 Search
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    metadatas = bm25_data["metadatas"]
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    bm25_results = [{"chunk": chunks[i], "metadata": metadatas[i]} for i in top_indices]

    # RRF 합산
    rrf_scores = {}
    chunks_map = {}
    for rank, r in enumerate(vector_results):
        key = r["chunk"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunks_map[key] = r
    for rank, r in enumerate(bm25_results):
        key = r["chunk"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunks_map[key] = r

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    fused = [chunks_map[k] for k in sorted_keys]

    # Re-ranking
    reranker = get_reranker()
    pairs = [[query, c["chunk"]] for c in fused[:20]]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(
        zip(fused[:20], rerank_scores),
        key=lambda x: x[1], reverse=True
    )[:top_k]

    final_chunks = [r[0]["chunk"] for r in reranked]
    context = "\n\n---\n\n".join(final_chunks)

    llm = OllamaLLM(model="qwen2.5:14b", timeout=180, num_predict=512)
    prompt = PromptTemplate(
        template="""Answer the question based only on the context below.
If the answer is not in the context, say 'I cannot find the information.'

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]""",
        input_variables=["context", "question"]
    )
    answer = (prompt | llm).invoke({"context": context, "question": query})
    return {"answer": answer, "contexts": final_chunks}


def safe_mean(val):
    if isinstance(val, list):
        clean = [x for x in val if x is not None and str(x) != 'nan']
        return float(np.mean(clean)) if clean else 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def run_evaluation(qa_path: str = "data/qa_dataset.json", sample_size: int = 10):
    # 데이터 로드
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    samples = random.sample(qa_data, min(sample_size, len(qa_data)))

    # 인덱스 로드
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("arxiv_papers")
    embeddings = OllamaEmbeddings(model="bge-m3")

    with open("data/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    # Vector RAG 답변 수집
    print("🔍 Vector RAG 평가 중...")
    vector_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, sample in enumerate(samples):
        print(f"  {i+1}/{len(samples)}")
        result = vector_ask(sample["question"], collection, embeddings)
        vector_data["question"].append(sample["question"])
        vector_data["answer"].append(result["answer"])
        vector_data["contexts"].append(result["contexts"])
        vector_data["ground_truth"].append(sample["answer"])

    # Hybrid RAG 답변 수집
    print("\n🔍 Hybrid RAG 평가 중...")
    hybrid_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, sample in enumerate(samples):
        print(f"  {i+1}/{len(samples)}")
        result = hybrid_ask_eval(sample["question"], collection, embeddings, bm25_data)
        hybrid_data["question"].append(sample["question"])
        hybrid_data["answer"].append(result["answer"])
        hybrid_data["contexts"].append(result["contexts"])
        hybrid_data["ground_truth"].append(sample["answer"])

    # RAGAS 설정: Ollama를 OpenAI 호환 API로 연결
    ollama_client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )

    ragas_llm = llm_factory(
        provider="openai",
        model="qwen2.5:14b",
        client=ollama_client
    )

    ragas_emb = embedding_factory(
        provider="openai",
        model="bge-m3",
        client=ollama_client
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=3
    )

    print("\n📊 RAGAS 평가 중 (시간이 걸려요)...")

    vector_score = evaluate(
        Dataset.from_dict(vector_data),
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config
    )

    hybrid_score = evaluate(
        Dataset.from_dict(hybrid_data),
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config
    )

    # 결과 비교표 출력
    metric_keys = {
        "Faithfulness":     "faithfulness",
        "AnswerRelevancy":  "answer_relevancy",
        "ContextPrecision": "context_precision",
        "ContextRecall":    "context_recall"
    }

    print("\n" + "=" * 60)
    print("📊 평가 결과 비교")
    print("=" * 60)
    print(f"{'지표':<25} {'Vector RAG':>12} {'Hybrid RAG':>12} {'개선':>8}")
    print("-" * 60)

    results = []
    for display_name, key in metric_keys.items():
        try:
            v = safe_mean(vector_score[key])
            h = safe_mean(hybrid_score[key])
        except Exception:
            v, h = 0.0, 0.0
        diff = h - v
        arrow = "↑" if diff > 0 else "↓"
        print(f"{display_name:<25} {v:>12.3f} {h:>12.3f} {arrow}{abs(diff):>6.3f}")
        results.append({"metric": key, "vector_rag": v, "hybrid_rag": h})

    print("=" * 60)

    # CSV 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/eval_results.csv", index=False)
    print("\n✅ 결과 저장 완료 → data/eval_results.csv")

    return vector_score, hybrid_score


if __name__ == "__main__":
    run_evaluation(sample_size=10)