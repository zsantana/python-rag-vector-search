import streamlit as st
from app.presentation.sidebar import sidebar
from app.presentation.chat import chat_interface, format_token_footer
from app.presentation.progress import progress_bar
from app.application.vectorization import process_uploaded_files
from app.application.search import get_search_strategy, search
from app.infrastructure.llm import get_llm_client


def _build_context(search_results, max_chars: int = 6000) -> str:
    if not search_results:
        return ""

    parts = []
    total = 0
    for i, r in enumerate(search_results, start=1):
        snippet = r.content.strip()
        block = f"[Trecho {i} | score={r.score:.4f}]\n{snippet}"
        if total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)

def main():
    st.title("RAG Vector V2")
    
    sidebar_configs = sidebar()
    chat_interface()
    
    if sidebar_configs["uploaded_files"]:
        if st.sidebar.button("Iniciar Vetorização"):
            progress, progress_text = progress_bar()
            process_uploaded_files(
                sidebar_configs["uploaded_files"],
                sidebar_configs["chunk_size"],
                sidebar_configs["overlap"],
                sidebar_configs["embedding_model"],
                progress,
                progress_text
            )

    if prompt := st.chat_input("Digite sua pergunta"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            input_tokens = output_tokens = total_tokens = None
            
            # Search for relevant documents
            search_strategy = get_search_strategy(
                sidebar_configs["search_type"],
                sidebar_configs["embedding_model"],
                sidebar_configs.get("vector_weight"),
                sidebar_configs.get("minimum_score")
            )
            search_results = search(prompt, search_strategy, int(sidebar_configs["top_k"]))
            context_text = _build_context(search_results)
            
            llm_client = get_llm_client()

            system_prompt = (
                "Você é um assistente útil. Use o CONTEXTO fornecido para responder. "
                "Se o CONTEXTO não for suficiente, diga claramente que não encontrou informação nos documentos.\n\n"
                f"CONTEXTO:\n{context_text if context_text else '[vazio]'}"
            )

            stream = llm_client.chat.completions.create(
                model=sidebar_configs["llm_model"],
                messages=(
                    [{"role": "system", "content": system_prompt}]
                    + [
                        {"role": m["role"], "content": m.get("content", "")}
                        for m in st.session_state.messages
                    ]
                ),
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)

            footer = format_token_footer(input_tokens, output_tokens, total_tokens)
            message_placeholder.markdown(f"{full_response}{footer}")

        st.session_state.messages.append({"role": "assistant", "content": full_response, "footer": footer})

if __name__ == "__main__":
    main()
