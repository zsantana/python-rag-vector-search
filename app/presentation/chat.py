import streamlit as st

def chat_interface():
    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message.get("content", "")
            footer = message.get("footer", "")
            st.markdown(f"{content}{footer}")


def format_token_footer(input_tokens: int | None, output_tokens: int | None, total_tokens: int | None) -> str:
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return ""
    return (
        "\n\n---\n"
        f"Tokens (entrada/saída/total): {input_tokens} / {output_tokens} / {total_tokens}"
    )
