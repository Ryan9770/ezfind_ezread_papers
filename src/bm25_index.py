import json
import pickle
import os
from rank_bm25 import BM25Okapi

def build_bm25_index(papers: list, save_path: str = "data/bm25_index.pkl"):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []
    all_metadatas = []

    for paper in papers:
        chunks = splitter.split_text(paper["text"])
        for i, chunk in enumerate(chunks):
            cleaned = chunk.strip()
            if len(cleaned) < 100:
                continue
            if not any(c.isalpha() for c in cleaned):
                continue
            if re.search(r'https?://\S+', cleaned) and len(cleaned) < 200:
                continue
            alpha_ratio = sum(c.isalpha() for c in cleaned) / len(cleaned)
            if alpha_ratio < 0.3:
                continue

            all_chunks.append(cleaned)
            all_metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "published": paper["published"],
                "chunk_index": i
            })

    # BM25는 토크나이즈된 리스트 필요
    tokenized = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized)

    # 저장
    os.makedirs("data", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": all_chunks,
            "metadatas": all_metadatas
        }, f)

    print(f"✅ BM25 인덱스 저장 완료 → {save_path} ({len(all_chunks)}개 청크)")
    return bm25, all_chunks, all_metadatas


if __name__ == "__main__":
    with open("data/processed/papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    build_bm25_index(papers)