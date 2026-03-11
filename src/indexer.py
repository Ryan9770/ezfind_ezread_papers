import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import json
import re

def build_index(
    papers: list,
    collection_name: str = "arxiv_papers",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    # ChromaDB 초기화
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 기존 컬렉션 있으면 삭제 후 재생성
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name)
    
    # 임베딩 모델
    embeddings = OllamaEmbeddings(model="bge-m3")
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    for paper in papers:
        chunks = splitter.split_text(paper["text"])
        
        for i, chunk in enumerate(chunks):

            cleaned = chunk.strip()

            # 1. 너무 짧은 청크 제거
            if len(cleaned) < 100:
                continue

            # 2. 알파벳이 하나도 없는 청크 제거 (숫자/특수문자만 있는 경우)
            if not any(c.isalpha() for c in cleaned):
                continue

            # 3. URL만 있거나 너무 짧은 청크 제거
            if re.search(r'https?://\S+', cleaned) and len(cleaned) < 200:
                continue

            # 4. 알파벳 비율 30% 미만 (수식, 기호)
            alpha_ratio = sum(c.isalpha() for c in cleaned) / len(cleaned)
            if alpha_ratio < 0.3:
                continue

            all_chunks.append(chunk)
            all_ids.append(f"{paper['id']}_{i}")
            all_metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "published": paper["published"],
                "chunk_index": i
            })
    
    print(f"총 {len(all_chunks)}개 청크 생성 중...")
    
    # 배치로 임베딩 & 저장 (ChromaDB 한번에 너무 많으면 느림)
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_ids = all_ids[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        
        batch_embeddings = embeddings.embed_documents(batch_chunks)
        
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metas
        )
        print(f"  {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} 완료")
    
    print(f"✅ ChromaDB 저장 완료 — 컬렉션: {collection_name}")
    return collection


if __name__ == "__main__":
    with open("data/processed/papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    build_index(papers)