import streamlit as st

def chat_interface():
    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_token_metrics(input_tokens, output_tokens, total_tokens):
    st.sidebar.subheader("Métricas de Tokens")
    st.sidebar.text(f"Tokens de entrada: {input_tokens}")
    st.sidebar.text(f"Tokens de saída: {output_tokens}")
    st.sidebar.text(f"Total de tokens: {total_tokens}")
