import streamlit as st
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download("punkt")

st.set_page_config(page_title="LangChain Chunking App", layout="wide")
st.title("📚 LangChain Chunking Practical")

# Upload text file
uploaded_file = st.file_uploader("📄 Upload a 5-page text file", type=["txt"])

# User queries
query_input = st.text_area("💬 Enter at least 5 user queries (one per line):", height=150)

# Chunking technique selector
chunk_method = st.selectbox(
    "🧠 Choose Chunking Method:",
    [
        "Fixed Size (Character-Based)",
        "Token-Based",
        "Recursive (Smart Chunking)",
        "Paragraph-Based (\\n\\n Separator)"
    ]
)

# Chunk size and overlap
chunk_size = st.slider("📏 Chunk Size", min_value=100, max_value=1000, step=100, value=300)
chunk_overlap = st.slider("🔁 Chunk Overlap", min_value=0, max_value=300, step=50, value=50)

# When user clicks the run button
if st.button("🚀 Run Chunking"):
    if not uploaded_file:
        st.warning("⚠️ Please upload a text file.")
    elif len([q.strip() for q in query_input.strip().split("\n") if q.strip()]) == 0:
        st.warning("⚠️ Please enter at least one query.")
    else:
        # Read and decode the uploaded text
        text = uploaded_file.read().decode("utf-8")

        # Select chunking method
        if chunk_method == "Fixed Size (Character-Based)":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=""  # Split at character level
            )
        elif chunk_method == "Token-Based":
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
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
            st.error("❌ Unknown chunking method selected.")
            st.stop()

        # Perform chunking
        chunks = splitter.split_text(text)

        # Show results
        st.subheader("📊 Chunking Results")
        st.success(f"✅ Total Chunks Generated: {len(chunks)}")

        st.subheader("🔍 Preview All Chunks")
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(chunk[:1000], language="text")  # Preview up to 1000 characters

        # Generate query list
        queries = [q.strip() for q in query_input.strip().split("\n") if q.strip()]

        # Semantic Similarity Evaluation
        st.subheader("🔍 Semantic Matching Results")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = model.encode(chunks)
        query_embeddings = model.encode(queries)

        for idx, query in enumerate(queries):
            st.markdown(f"### 💬 Query {idx+1}: `{query}`")

            # Compute cosine similarity with all chunks
            scores = cosine_similarity([query_embeddings[idx]], chunk_embeddings)[0]
            top_indices = np.argsort(scores)[::-1][:3]

            # Display top 3 chunks
            for rank, i in enumerate(top_indices):
                st.markdown(f"**Top {rank+1} Match (Score: {scores[i]:.2f})**")
                st.code(chunks[i][:500], language="text")  # Show top 500 chars

            # Simple evaluation for context retention
            if scores[top_indices[0]] - scores[top_indices[1]] > 0.15:
                retention = "✅ High context retention (one chunk is clearly most relevant)"
            else:
                retention = "⚠️ Context may be spread across multiple chunks"

            st.markdown(f"**🧠 Evaluation:** {retention}")
            st.markdown("---")
