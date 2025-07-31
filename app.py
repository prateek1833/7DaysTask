import streamlit as st
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import nltk

nltk.download("punkt")

st.set_page_config(page_title="RAG Performance with FAISS", layout="wide")
st.title("🔍 RAG Evaluation: Chunking + FAISS + Prompting Techniques")

# Upload section
uploaded_file = st.file_uploader("📄 Upload a 5-page text file", type=["txt"])

# Chunking settings
chunk_method = st.selectbox("📎 Choose Chunking Method:", [
    "Fixed Size (Character-Based)",
    "Token-Based",
    "Recursive (Smart Chunking)",
    "Paragraph-Based (\\n\\n Separator)"
])
chunk_size = st.slider("📏 Chunk Size", min_value=100, max_value=1000, step=100, value=300)
chunk_overlap = st.slider("🔁 Chunk Overlap", min_value=0, max_value=300, step=50, value=50)

# Embedding model (shared)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Run logic
if st.button("🚀 Run Chunking and FAISS Indexing"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a text file.")
        st.stop()

    # Read text
    text = uploaded_file.read().decode("utf-8")

    # Chunking logic
    if chunk_method == "Fixed Size (Character-Based)":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunk_method == "Token-Based":
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunk_method == "Recursive (Smart Chunking)":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", "\n", " "]
        )
    elif chunk_method == "Paragraph-Based (\\n\\n Separator)":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        st.error("Invalid chunking method selected.")
        st.stop()

    chunks = splitter.split_text(text)
    st.session_state.chunks = chunks
    st.success(f"✅ Total Chunks Generated: {len(chunks)}")

    chunk_embeddings = model.encode(chunks)
    st.session_state.chunk_embeddings = chunk_embeddings

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype("float32"))
    st.session_state.index = index

# Evaluation section
if "chunks" in st.session_state and "chunk_embeddings" in st.session_state and "index" in st.session_state:
    query_input = st.text_area("💬 Enter user queries (one per line):", height=150)

    if st.button("🔎 Run Evaluation"):
        queries = [q.strip() for q in query_input.strip().split("\n") if q.strip()]
        if len(queries) == 0:
            st.warning("⚠️ Please enter queries.")
            st.stop()

        query_embeddings = model.encode(queries)
        st.subheader("📌 Top 3 Matching Chunks per Query (via FAISS)")
        similarities = []

        for idx, query in enumerate(queries):
            st.markdown(f"### 🔹 Query {idx+1}: `{query}`")

            query_vector = np.array([query_embeddings[idx]]).astype("float32")
            D, I = st.session_state.index.search(query_vector, k=3)
            top_indices = I[0]
            top_matches = [st.session_state.chunks[i] for i in top_indices]

            for rank, i in enumerate(top_indices):
                st.markdown(f"**Top {rank+1} Match (FAISS L2 Distance: {D[0][rank]:.2f})**")
                st.code(st.session_state.chunks[i][:400], language="text")

            # Always use Zero-Shot prompting (just use top-1 match as response)
            response = top_matches[0]
            st.markdown(f"**📤 Simulated Answer:** {response[:400]}...")
            st.markdown("---")

            # Semantic similarity evaluation
            response_embedding = model.encode([response])[0]
            similarity_score = cosine_similarity(
                [query_embeddings[idx]],
                [response_embedding]
            )[0][0]
            similarities.append(similarity_score)

        average_similarity = np.mean(similarities)
        st.metric("🧠 Average Semantic Similarity", f"{average_similarity:.2f}")
