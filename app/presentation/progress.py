import streamlit as st

class _NoOpText:
    def text(self, *_args, **_kwargs):
        return None


def progress_bar(*, in_sidebar: bool = False, show_header: bool = True, show_text: bool = True):
    container = st.sidebar if in_sidebar else st
    with container:
        if show_header:
            st.header("Progresso da Vetorização")
        progress = st.progress(0)
        progress_text = st.empty() if show_text else _NoOpText()
    return progress, progress_text
