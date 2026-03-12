import pickle
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

PROMPT_TEMPLATE = """
아래 논문 내용을 바탕으로 질문에 답해줘.
답변은 반드시 제공된 컨텍스트에 근거해야 해.
컨텍스트에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 해줘.

[컨텍스트]
{context}

[질문]
{question}

[답변]
"""

def load_indexes():
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("arxiv_papers")
    embeddings = OllamaEmbeddings(model="bge-m3")

    # BM25
    with open("data/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    return collection, embeddings, bm25_data


def vector_search(query: str, collection, embeddings, top_k: int = 10):
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "chunk": doc,
            "metadata": meta,
            "score": 1 - dist  # 거리 → 유사도로 변환
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def bm25_search(query: str, bm25_data: dict, top_k: int = 10):
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    metadatas = bm25_data["metadatas"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # 상위 top_k 인덱스
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "chunk": chunks[i],
            "metadata": metadatas[i],
            "score": scores[i]
        }
        for i in top_indices
    ]


def reciprocal_rank_fusion(vector_results: list, bm25_results: list, k: int = 60):
    """RRF: 두 검색 결과를 순위 기반으로 합산"""
    scores = {}
    chunks_map = {}

    for rank, result in enumerate(vector_results):
        key = result["chunk"][:100]  # 앞 100자를 키로 사용
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks_map[key] = result

    for rank, result in enumerate(bm25_results):
        key = result["chunk"][:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks_map[key] = result

    # RRF 점수로 정렬
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        {**chunks_map[key], "rrf_score": scores[key]}
        for key in sorted_keys
    ]


def rerank(query: str, candidates: list, top_k: int = 5):
    """CrossEncoder로 최종 Re-ranking"""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, c["chunk"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # 점수 붙여서 정렬
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def hybrid_ask(query: str, top_k: int = 5) -> dict:
    collection, embeddings, bm25_data = load_indexes()

    # 1. Vector Search
    vector_results = vector_search(query, collection, embeddings, top_k=10)

    # 2. BM25 Search
    bm25_results = bm25_search(query, bm25_data, top_k=10)

    # 3. RRF 합산
    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    # 4. Re-ranking
    reranked = rerank(query, fused[:20], top_k=top_k)

    # 5. LLM 답변 생성
    context = "\n\n---\n\n".join([r["chunk"] for r in reranked])
    llm = OllamaLLM(model="llama3.2")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})

    return {
        "answer": answer,
        "sources": reranked
    }


if __name__ == "__main__":
    result = hybrid_ask("What is RAG and how does it work?")

    print("=" * 60)
    print("답변:", result["answer"])
    print("\n출처:")
    for s in result["sources"]:
        print(f"  - {s['metadata']['title']}")
        print(f"    RRF: {s['rrf_score']:.4f} | Rerank: {s['rerank_score']:.3f}")