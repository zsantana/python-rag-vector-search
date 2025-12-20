import os
import zipfile
import tempfile
import io
from typing import Callable, Optional

import tiktoken

from app.domain.chunking import chunk_text
from app.infrastructure.embeddings import get_embedding_model
from app.infrastructure.database import insert_document, create_table
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

try:
    # Newer LangChain splits callbacks into community
    from langchain_community.callbacks.manager import get_openai_callback  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Older LangChain location
        from langchain.callbacks import get_openai_callback  # type: ignore
    except Exception:  # pragma: no cover
        get_openai_callback = None

def _read_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    return loader.load()


def _count_tokens(text: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def _process_file(
    file_path: str,
    display_name: str,
    embedding_model,
    embedding_model_name: str,
    chunk_size: int,
    overlap: int,
    log_fn: Optional[Callable[[str], None]] = None,
):
    documents = _read_document(file_path)
    for doc_i, document in enumerate(documents, start=1):
        chunks = chunk_text(document.page_content, chunk_size, overlap)
        if log_fn:
            log_fn(
                f"**Arquivo:** `{display_name}` | **Doc:** {doc_i}/{len(documents)} | **Chunks:** {len(chunks)}"
            )

        chunk_token_counts = [
            _count_tokens(chunk, embedding_model_name)
            for chunk in chunks
        ]

        embedding_total_tokens = None
        if get_openai_callback is not None:
            try:
                with get_openai_callback() as cb:
                    embeddings = embedding_model.embed_documents(chunks)
                embedding_total_tokens = getattr(cb, "total_tokens", None)
            except Exception:
                embeddings = embedding_model.embed_documents(chunks)
        else:
            embeddings = embedding_model.embed_documents(chunks)

        running_tokens = 0
        for i, chunk in enumerate(chunks):
            insert_document(chunk, embeddings[i])
            running_tokens += chunk_token_counts[i]
            if log_fn:
                snippet = " ".join(chunk.strip().split())[:120]
                snippet = snippet.replace("`", "\\`")
                log_fn(
                    "- "
                    f"**Chunk {i + 1}/{len(chunks)}** — "
                    f"tokens: `{chunk_token_counts[i]}` (Σ `{running_tokens}`) "
                    f"— _{snippet}_"
                )

        if log_fn:
            approx_total = sum(chunk_token_counts)
            if embedding_total_tokens is not None:
                log_fn(
                    f"> **Resumo embeddings:** total_tokens (API) = `{embedding_total_tokens}` | tokens (aprox) = `{approx_total}`"
                )
            else:
                log_fn(f"> **Resumo embeddings:** tokens (aprox) = `{approx_total}`")


def process_uploaded_files(
    uploaded_files,
    chunk_size,
    overlap,
    embedding_model_name,
    progress_bar,
    progress_text,
    log_fn: Optional[Callable[[str], None]] = None,
):
    create_table()

    # Progresso: conta cada arquivo efetivamente processado.
    # - Arquivo normal: 1 unidade
    # - ZIP: cada arquivo dentro do ZIP (não diretório) conta como 1 unidade
    def _count_zip_members(uploaded_file) -> int:
        try:
            data = uploaded_file.getbuffer()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                return sum(1 for info in zf.infolist() if not info.is_dir())
        except Exception:
            return 1

    total_units = 0
    for f in uploaded_files:
        ext = os.path.splitext(getattr(f, "name", ""))[1].lower()
        total_units += _count_zip_members(f) if ext == ".zip" else 1
    if total_units <= 0:
        total_units = 1

    embedding_model = get_embedding_model(embedding_model_name)
    processed_units = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == ".zip":
                zip_ref = zipfile.ZipFile(file_path, "r")
                extracted_files_dir = os.path.join(temp_dir, "extracted")
                zip_ref.extractall(extracted_files_dir)
                zip_ref.close()
                for root, _, files in os.walk(extracted_files_dir):
                    for file in files:
                        extracted_path = os.path.join(root, file)
                        _process_file(
                            extracted_path,
                            display_name=f"{uploaded_file.name}::{file}",
                            embedding_model=embedding_model,
                            embedding_model_name=embedding_model_name,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            log_fn=log_fn,
                        )
                        processed_units += 1
                        progress_bar.progress(processed_units / total_units)
                        progress_text.text(
                            f"Processando arquivo {processed_units}/{total_units}: {uploaded_file.name}::{file}"
                        )
            else:
                _process_file(
                    file_path,
                    display_name=uploaded_file.name,
                    embedding_model=embedding_model,
                    embedding_model_name=embedding_model_name,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    log_fn=log_fn,
                )

                processed_units += 1
                progress_bar.progress(processed_units / total_units)
                progress_text.text(
                    f"Processando arquivo {processed_units}/{total_units}: {uploaded_file.name}"
                )
    progress_text.text("Processamento concluído.")