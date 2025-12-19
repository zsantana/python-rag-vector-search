import streamlit as st

def progress_bar():
    st.header("Progresso da Vetorização")
    progress_bar = st.progress(0)
    progress_text = st.empty()
    return progress_bar, progress_text
