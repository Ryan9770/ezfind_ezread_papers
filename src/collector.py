""" ArXiv 논문 수집기 """

import arxiv
import os
from tqdm import tqdm

def collect_papers(query: str, max_results: int = 50, download_dir: str = "data/pdfs"):
    os.makedirs(download_dir, exist_ok=True)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for paper in tqdm(client.results(search), desc="Downloading papers"):
        try:
            filename = f"{paper.get_short_id().replace('/', '_')}.pdf"
            filepath = os.path.join(download_dir, filename)
            
            if not os.path.exists(filepath):
                paper.download_pdf(dirpath=download_dir, filename=filename)
            
            papers.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "abstract": paper.summary,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published),
                "filepath": filepath
            })
        except Exception as e:
            print(f"Error downloading {paper.title}: {e}")
    
    print(f"\n✅ {len(papers)}개 논문 수집 완료")
    return papers


if __name__ == "__main__":
    papers = collect_papers(
        query="RAG retrieval augmented generation",
        max_results=30
    )
    for p in papers[:3]:
        print(f"- {p['title']}")