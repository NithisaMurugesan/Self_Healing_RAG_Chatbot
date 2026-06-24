import streamlit as st
from rag import build_rag, ask_question

st.set_page_config(
    page_title="Self-Healing RAG Chatbot",
    page_icon="📖",
    layout="wide"
)

st.title("🤖 Self-Healing RAG Chatbot")

uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

# File upload 
if uploaded_file:
        with st.spinner("Processing PDF and building knowledge base..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            (vector_store,compression_retriever) = build_rag("temp.pdf")

            st.session_state.vector_store = (vector_store)

            st.session_state.compression_retriever = (compression_retriever)

        st.success("✅ PDF processed successfully!")

# Chat box
if "vector_store" in st.session_state:

    question = st.text_input( "💬 Ask a question about the uploaded PDF")
    if question:
        with st.spinner("Generating your answer..."):
            result = ask_question(question, st.session_state.vector_store, st.session_state.compression_retriever)

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(result["answer"])

        st.divider()

        # Monitoring Metrics
        total_time = (
            result["rewrite_time"]
            + result["retrieval_time"]
            + result["generation_time"]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Faithfulness",result["faithfulness_score"])

        with col2:
            st.metric("Retries", result["retry_count"])

        with col3:
            st.metric("Latency",f"{total_time:.2f}s")

        with col4:
            st.metric("Chunks",3)

        # Monitoring Details
        with st.expander("🔍 Monitoring & Evaluation"):
            st.subheader("Query Rewriting")
            st.write(result["rewritten_question"])

            st.subheader("Evaluation")
            st.write(f"Faithfulness Score: {result['faithfulness_score']}")
            st.write(result["faithfulness_reason"])

            st.subheader("Latency Breakdown")
            st.write(f"Rewrite Time: {result['rewrite_time']:.3f}s")
            st.write(f"Retrieval Time: {result['retrieval_time']:.3f}s")
            st.write(f"Generation Time: {result['generation_time']:.3f}s")
            st.write( f"Total Time: {total_time:.3f}s")