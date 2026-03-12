import json
import random
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

PROMPT = """
아래 논문 텍스트를 읽고, 이 텍스트로 답할 수 있는 질문과 답변을 JSON 형식으로 만들어줘.

규칙:
- 질문은 구체적이고 명확하게
- 답변은 텍스트에서 직접 찾을 수 있는 내용으로
- 반드시 아래 JSON 형식만 출력 (다른 말 하지 말 것)

형식:
{{"question": "질문 내용", "answer": "답변 내용"}}

[텍스트]
{context}
"""

def generate_qa_dataset(
    papers: list,
    num_samples: int = 50,
    save_path: str = "data/qa_dataset.json"
):
    llm = OllamaLLM(model="llama3.2")
    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["context"]
    )
    chain = prompt | llm

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    # 모든 청크 수집
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
            if sum(c.isalpha() for c in cleaned) / len(cleaned) < 0.3:
                continue
            all_chunks.append(cleaned)
            all_metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"]
            })

    # 랜덤 샘플링
    indices = random.sample(range(len(all_chunks)), min(num_samples, len(all_chunks)))

    qa_dataset = []
    failed = 0

    for idx, i in enumerate(indices):
        chunk = all_chunks[i]
        meta = all_metadatas[i]

        try:
            response = chain.invoke({"context": chunk})

            # JSON 파싱
            # 응답에서 { } 부분만 추출
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                failed += 1
                continue

            qa = json.loads(response[start:end])

            if "question" not in qa or "answer" not in qa:
                failed += 1
                continue

            qa_dataset.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "context": chunk,
                "source_title": meta["title"],
                "paper_id": meta["paper_id"]
            })

            print(f"  [{idx+1}/{len(indices)}] ✅ {meta['title'][:50]}...")

        except Exception as e:
            failed += 1
            print(f"  [{idx+1}/{len(indices)}] ❌ 실패: {e}")

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ QA 데이터셋 생성 완료: {len(qa_dataset)}개 (실패: {failed}개)")
    print(f"저장 위치: {save_path}")
    return qa_dataset


if __name__ == "__main__":
    with open("data/processed/papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    generate_qa_dataset(papers, num_samples=50)