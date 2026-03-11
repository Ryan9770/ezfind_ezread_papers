import fitz  # pymupdf
import json
import os

def parse_pdf(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def parse_all_papers(papers: list, save_dir: str = "data/processed") -> list:
    os.makedirs(save_dir, exist_ok=True)
    
    parsed = []
    for paper in papers:
        try:
            text = parse_pdf(paper["filepath"])
            
            # 너무 짧은 건 스킵 (파싱 실패)
            if len(text) < 500:
                print(f"⚠️ 스킵 (텍스트 너무 짧음): {paper['title']}")
                continue
            
            parsed.append({
                **paper,
                "text": text
            })
            
        except Exception as e:
            print(f"❌ 파싱 실패 {paper['title']}: {e}")
    
    # 저장
    save_path = os.path.join(save_dir, "papers.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {len(parsed)}개 파싱 완료 → {save_path}")
    return parsed


if __name__ == "__main__":
    from collector import collect_papers
    papers = collect_papers("RAG retrieval augmented generation", max_results=30)
    parsed = parse_all_papers(papers)