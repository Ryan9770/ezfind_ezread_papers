import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

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

def get_retriever(collection_name: str = "arxiv_papers"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)
    embeddings = OllamaEmbeddings(model="bge-m3")
    return collection, embeddings

def retrieve(query: str, collection, embeddings, top_k: int = 5):
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    return [
        {"chunk": c, "metadata": m, "distance": d}
        for c, m, d in zip(chunks, metadatas, distances)
    ]

def ask(query: str, collection, embeddings, top_k: int = 5) -> dict:
    # 검색
    retrieved = retrieve(query, collection, embeddings, top_k)
    context = "\n\n---\n\n".join([r["chunk"] for r in retrieved])
    
    # LLM 답변
    llm = OllamaLLM(model="llama3.2")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})
    
    return {
        "answer": answer,
        "sources": retrieved
    }


if __name__ == "__main__":
    collection, embeddings = get_retriever()
    result = ask("What is RAG and how does it work?", collection, embeddings)
    print("답변:", result["answer"])
    print("\n출처:")
    for s in result["sources"]:
        print(f"  - {s['metadata']['title']} (거리: {s['distance']:.3f})")