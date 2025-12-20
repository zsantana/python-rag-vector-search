import os

import streamlit as st
from app.infrastructure.llm import get_available_gpt_models
from app.infrastructure.database import truncate_documents_table

def sidebar():
    with st.sidebar:
        st.title("Configurações")

        if st.button("Resetar conversa"):
            st.session_state.pop("messages", None)
            st.rerun()

        if st.button("Truncate vetorização (limpar tabela)"):
            try:
                truncate_documents_table()
                st.success("Tabela de vetorização limpa com sucesso.")
            except Exception as e:
                st.error(f"Falha ao limpar tabela: {e}")

        # 1.1 Upload de Arquivos
        uploaded_files = st.file_uploader(
            "Upload de Arquivos",
            type=["pdf", "docx", "txt", "md", "zip"],
            accept_multiple_files=True
        )

        # 1.2 Seleção de Modelo LLM
        gpt_models = get_available_gpt_models()
        default_llm_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")
        default_llm_index = gpt_models.index(default_llm_model) if default_llm_model in gpt_models else 0
        llm_model = st.selectbox(
            "Modelo LLM",
            gpt_models,
            index=default_llm_index,
        )

        # 1.3 Seleção de Modelo de Embeddings
        embedding_model = st.selectbox(
            "Modelo de Embedding",
            ["text-embedding-ada-002", "text-embedding-3-small"]
        )

        # 1.4 Tipo de Busca
        search_type = st.selectbox(
            "Tipo de Busca",
            ["Vetorial", "Semântica", "Híbrida"]
        )

        # 1.5 Parâmetros de Vetorização
        st.subheader("Parâmetros de Vetorização")
        chunk_size = st.number_input("Chunk Size", value=1000)
        overlap = st.number_input("Overlap", value=200)
        top_k = st.number_input("Top K", value=10)

        # 1.6 Parâmetros Específicos da Busca Híbrida
        if search_type == "Híbrida":
            st.subheader("Parâmetros da Busca Híbrida")
            vector_weight = st.slider("Vector Weight", 0.0, 1.0, 0.70)
            minimum_score = st.slider("Minimum Score", 0.0, 1.0, 0.30)
        else:
            vector_weight = 0.70
            minimum_score = 0.30

    return {
        "uploaded_files": uploaded_files,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "search_type": search_type,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "vector_weight": vector_weight,
        "minimum_score": minimum_score
    }
