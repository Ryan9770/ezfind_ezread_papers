# debug_chunks.py 로 저장 후 실행
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

with open("data/processed/papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

all_chunks = []
for paper in papers:
    chunks = splitter.split_text(paper["text"])
    for i, chunk in enumerate(chunks):
        cleaned = chunk.strip()
        if len(cleaned) < 50:
            continue
        if not any(c.isalpha() for c in cleaned):
            continue
        all_chunks.append(cleaned)

print(f"총 청크 수: {len(all_chunks)}")

# 1950~2000번째 청크 근처를 1개씩 임베딩 테스트
embeddings = OllamaEmbeddings(model="bge-m3")

for i in range(1900, min(2000, len(all_chunks))):
    try:
        embeddings.embed_documents([all_chunks[i]])
    except Exception as e:
        print(f"❌ 문제 청크 인덱스: {i}")
        print(f"내용 앞 200자: {all_chunks[i][:200]}")
        print(f"에러: {e}")