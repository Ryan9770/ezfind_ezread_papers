import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import chromadb
import plotly.graph_objects as go
import plotly.express as px
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 스타일 ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* 배경 */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e3a;
}

/* 헤더 */
.dash-header {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7c6af7;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.dash-sub {
    font-size: 0.9rem;
    color: #5a5a7a;
    font-family: 'Space Mono', monospace;
    margin-bottom: 2rem;
}

/* 메트릭 카드 */
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.75rem;
    color: #5a5a7a;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    color: #e8e8f0;
}
.metric-delta-up { color: #4ade80; font-size: 0.85rem; }
.metric-delta-down { color: #f87171; font-size: 0.85rem; }

/* 답변 박스 */
.answer-box {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-left: 3px solid #7c6af7;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c8c8e0;
}
.hybrid-box {
    border-left: 3px solid #4ade80;
}

/* 소스 뱃지 */
.source-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #7c6af7;
    font-family: 'Space Mono', monospace;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* 탭 */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #5a5a7a;
}
.stTabs [aria-selected="true"] {
    color: #7c6af7 !important;
}

/* 입력창 */
.stTextInput > div > div > input {
    background: #0f0f1a;
    border: 1px solid #2a2a4a;
    color: #e8e8f0;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
}

/* 버튼 */
.stButton > button {
    background: #7c6af7;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: #6a58e5;
}

/* 구분선 */
hr { border-color: #1e1e3a; }

/* 섹션 타이틀 */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e3a;
}
</style>
""", unsafe_allow_html=True)


# ── 캐시된 리소스 로드 ─────────────────────────────────────
@st.cache_resource
def load_indexes():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("arxiv_papers")
    emb_model = OllamaEmbeddings(model="bge-m3")
    with open("data/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return collection, emb_model, bm25_data, reranker

@st.cache_data
def load_eval_results():
    try:
        return pd.read_csv("data/eval_results.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_qa_dataset():
    try:
        with open("data/qa_dataset.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ── RAG 함수 ───────────────────────────────────────────────
PROMPT = """Answer the question based only on the context below.
If the answer is not in the context, say 'I cannot find the information.'

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]"""

def vector_search(query, collection, emb_model, top_k=5):
    qe = emb_model.embed_query(query)
    results = collection.query(
        query_embeddings=[qe],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]

def hybrid_search(query, collection, emb_model, bm25_data, reranker, top_k=5):
    # Vector
    qe = emb_model.embed_query(query)
    v = collection.query(query_embeddings=[qe], n_results=10, include=["documents","metadatas"])
    vector_results = [{"chunk": d, "metadata": m} for d, m in zip(v["documents"][0], v["metadatas"][0])]

    # BM25
    scores = bm25_data["bm25"].get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    bm25_results = [{"chunk": bm25_data["chunks"][i], "metadata": bm25_data["metadatas"][i]} for i in top_idx]

    # RRF
    rrf, cmap = {}, {}
    for rank, r in enumerate(vector_results):
        k = r["chunk"][:100]
        rrf[k] = rrf.get(k, 0) + 1/(61+rank)
        cmap[k] = r
    for rank, r in enumerate(bm25_results):
        k = r["chunk"][:100]
        rrf[k] = rrf.get(k, 0) + 1/(61+rank)
        cmap[k] = r

    fused = [cmap[k] for k in sorted(rrf, key=lambda x: rrf[x], reverse=True)]

    # Rerank
    pairs = [[query, c["chunk"]] for c in fused[:20]]
    rscores = reranker.predict(pairs)
    reranked = sorted(zip(fused[:20], rscores), key=lambda x: x[1], reverse=True)[:top_k]

    chunks = [r[0]["chunk"] for r in reranked]
    metas = [r[0]["metadata"] for r in reranked]
    rrank_scores = [float(r[1]) for r in reranked]
    return chunks, metas, rrank_scores

def generate_answer(query, chunks):
    context = "\n\n---\n\n".join(chunks)
    llm = OllamaLLM(model="qwen2.5:14b", timeout=120, num_predict=512)
    prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
    return (prompt | llm).invoke({"context": context, "question": query})


# ── 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="dash-header">⚡ RAG<br/>EVAL</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">arxiv · hybrid · rerank</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["📊 Evaluation Results", "🔍 Live QA", "📚 Dataset Browser"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-title">System</div>', unsafe_allow_html=True)

    eval_df = load_eval_results()
    qa_data = load_qa_dataset()

    if eval_df is not None:
        st.markdown(f"<span style='color:#4ade80; font-family:Space Mono; font-size:0.75rem'>✓ eval_results.csv</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#f87171; font-family:Space Mono; font-size:0.75rem'>✗ eval_results.csv 없음</span>", unsafe_allow_html=True)

    st.markdown(f"<span style='color:#7c6af7; font-family:Space Mono; font-size:0.75rem'>QA samples: {len(qa_data)}</span>", unsafe_allow_html=True)


# ── 페이지 1: 평가 결과 ────────────────────────────────────
if page == "📊 Evaluation Results":
    st.markdown('<div class="dash-header">Evaluation Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">Vector RAG vs Hybrid RAG (BM25 + Rerank)</div>', unsafe_allow_html=True)

    if eval_df is None:
        st.error("data/eval_results.csv 파일이 없어. evaluator.py를 먼저 실행해줘.")
    else:
        metric_labels = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall"
        }
        metric_desc = {
            "faithfulness": "답변이 컨텍스트에 근거한 정도",
            "answer_relevancy": "답변이 질문과 관련된 정도",
            "context_precision": "검색된 문서의 정확도",
            "context_recall": "필요한 문서를 얼마나 커버했는지"
        }

        # 메트릭 카드
        cols = st.columns(4)
        for i, (key, label) in enumerate(metric_labels.items()):
            row = eval_df[eval_df["metric"] == key]
            if row.empty:
                continue
            v = float(row["vector_rag"].values[0])
            h = float(row["hybrid_rag"].values[0])
            diff = h - v
            with cols[i]:
                arrow = "▲" if diff >= 0 else "▼"
                delta_class = "metric-delta-up" if diff >= 0 else "metric-delta-down"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{h:.3f}</div>
                    <div class="{delta_class}">{arrow} {abs(diff):.3f} vs Vector</div>
                    <div style="color:#3a3a5a; font-size:0.7rem; margin-top:0.4rem">{metric_desc[key]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 레이더 차트 + 바 차트
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="section-title">Radar Chart</div>', unsafe_allow_html=True)
            categories = [metric_labels[k] for k in metric_labels]
            vector_vals = [float(eval_df[eval_df["metric"]==k]["vector_rag"].values[0]) for k in metric_labels]
            hybrid_vals = [float(eval_df[eval_df["metric"]==k]["hybrid_rag"].values[0]) for k in metric_labels]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vector_vals + [vector_vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Vector RAG',
                line=dict(color='#7c6af7', width=2),
                fillcolor='rgba(124,106,247,0.15)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=hybrid_vals + [hybrid_vals[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Hybrid RAG',
                line=dict(color='#4ade80', width=2),
                fillcolor='rgba(74,222,128,0.15)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#0f0f1a',
                    radialaxis=dict(visible=True, range=[0,1], color='#3a3a5a', gridcolor='#1e1e3a'),
                    angularaxis=dict(color='#5a5a7a', gridcolor='#1e1e3a')
                ),
                paper_bgcolor='#0a0a0f',
                plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0', family='Space Mono'),
                legend=dict(bgcolor='#0f0f1a', bordercolor='#1e1e3a', borderwidth=1),
                margin=dict(l=60, r=60, t=40, b=40),
                height=350
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Score Comparison</div>', unsafe_allow_html=True)
            fig_bar = go.Figure()
            x_labels = [metric_labels[k] for k in metric_labels]
            fig_bar.add_trace(go.Bar(
                name='Vector RAG',
                x=x_labels,
                y=vector_vals,
                marker_color='#7c6af7',
                opacity=0.85
            ))
            fig_bar.add_trace(go.Bar(
                name='Hybrid RAG',
                x=x_labels,
                y=hybrid_vals,
                marker_color='#4ade80',
                opacity=0.85
            ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='#0a0a0f',
                plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0', family='Space Mono', size=11),
                xaxis=dict(gridcolor='#1e1e3a', color='#5a5a7a'),
                yaxis=dict(gridcolor='#1e1e3a', color='#5a5a7a', range=[0, 1]),
                legend=dict(bgcolor='#0f0f1a', bordercolor='#1e1e3a', borderwidth=1),
                margin=dict(l=40, r=20, t=40, b=40),
                height=350
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 상세 테이블
        st.markdown('<div class="section-title">Detail Table</div>', unsafe_allow_html=True)
        display_df = eval_df.copy()
        display_df["improvement"] = display_df["hybrid_rag"] - display_df["vector_rag"]
        display_df["metric"] = display_df["metric"].map(metric_labels)
        display_df.columns = ["Metric", "Vector RAG", "Hybrid RAG", "Improvement"]
        display_df = display_df.round(4)
        st.dataframe(
            display_df.style
                .format({"Vector RAG": "{:.4f}", "Hybrid RAG": "{:.4f}", "Improvement": "{:+.4f}"})
                .map(lambda v: "color: #4ade80" if v > 0 else "color: #f87171", subset=["Improvement"]),
            use_container_width=True,
            hide_index=True
        )


# ── 페이지 2: 실시간 QA ────────────────────────────────────
elif page == "🔍 Live QA":
    st.markdown('<div class="dash-header">Live QA</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">Vector vs Hybrid — 실시간 비교</div>', unsafe_allow_html=True)

    query = st.text_input("질문을 입력하세요", placeholder="What is RAG and how does it work?")

    col_opt1, col_opt2, _ = st.columns([1, 1, 4])
    with col_opt1:
        top_k = st.selectbox("Top-K", [3, 5, 7, 10], index=1)
    with col_opt2:
        mode = st.selectbox("Mode", ["Both", "Vector only", "Hybrid only"])

    run_btn = st.button("⚡ RUN")

    if run_btn and query:
        collection, emb_model, bm25_data, reranker = load_indexes()

        col1, col2 = st.columns(2)

        if mode in ["Both", "Vector only"]:
            with col1:
                st.markdown('<div class="section-title">🟣 Vector RAG</div>', unsafe_allow_html=True)
                with st.spinner("검색 중..."):
                    v_chunks, v_metas, v_dists = vector_search(query, collection, emb_model, top_k)
                with st.spinner("답변 생성 중..."):
                    v_answer = generate_answer(query, v_chunks)
                st.markdown(f'<div class="answer-box">{v_answer}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title" style="margin-top:1rem">Sources</div>', unsafe_allow_html=True)
                for i, (meta, dist) in enumerate(zip(v_metas, v_dists)):
                    with st.expander(f"[{i+1}] {meta.get('title','')[:60]}... (dist: {dist:.3f})"):
                        st.write(v_chunks[i][:400] + "...")

        if mode in ["Both", "Hybrid only"]:
            with col2:
                st.markdown('<div class="section-title">🟢 Hybrid RAG</div>', unsafe_allow_html=True)
                with st.spinner("BM25 + Vector + Rerank 중..."):
                    h_chunks, h_metas, h_scores = hybrid_search(query, collection, emb_model, bm25_data, reranker, top_k)
                with st.spinner("답변 생성 중..."):
                    h_answer = generate_answer(query, h_chunks)
                st.markdown(f'<div class="answer-box hybrid-box">{h_answer}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title" style="margin-top:1rem">Sources</div>', unsafe_allow_html=True)
                for i, (meta, score) in enumerate(zip(h_metas, h_scores)):
                    with st.expander(f"[{i+1}] {meta.get('title','')[:60]}... (rerank: {score:.3f})"):
                        st.write(h_chunks[i][:400] + "...")


# ── 페이지 3: 데이터셋 브라우저 ───────────────────────────
elif page == "📚 Dataset Browser":
    st.markdown('<div class="dash-header">Dataset Browser</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">자동 생성된 QA 데이터셋</div>', unsafe_allow_html=True)

    qa_data = load_qa_dataset()

    if not qa_data:
        st.error("data/qa_dataset.json 파일이 없어.")
    else:
        # 통계
        col1, col2, col3 = st.columns(3)
        papers = list(set(q["source_title"] for q in qa_data))
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total QA Pairs</div><div class="metric-value">{len(qa_data)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Unique Papers</div><div class="metric-value">{len(papers)}</div></div>', unsafe_allow_html=True)
        with col3:
            avg_q_len = int(np.mean([len(q["question"]) for q in qa_data]))
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Q Length</div><div class="metric-value">{avg_q_len}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # 검색 필터
        search_term = st.text_input("검색", placeholder="키워드로 필터링...")

        filtered = qa_data
        if search_term:
            filtered = [q for q in qa_data if search_term.lower() in q["question"].lower() or search_term.lower() in q["answer"].lower()]

        st.markdown(f'<div class="section-title">{len(filtered)}개 결과</div>', unsafe_allow_html=True)

        for i, qa in enumerate(filtered):
            with st.expander(f"Q{i+1}. {qa['question'][:80]}..."):
                st.markdown(f"**📄 논문:** `{qa['source_title'][:70]}`")
                st.markdown(f"**❓ 질문:** {qa['question']}")
                st.markdown(f"**✅ 정답:** {qa['answer']}")
                st.markdown(f"**📝 컨텍스트:**")
                st.text(qa["context"][:300] + "...")