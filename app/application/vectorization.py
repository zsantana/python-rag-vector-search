import os
import zipfile
import tempfile
import shutil
from app.domain.chunking import chunk_text
from app.infrastructure.embeddings import get_embedding_model
from app.infrastructure.database import insert_document, create_table
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

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

def _process_file(file_path, embedding_model, chunk_size, overlap):
    documents = _read_document(file_path)
    for document in documents:
        chunks = chunk_text(document.page_content, chunk_size, overlap)
        embeddings = embedding_model.embed_documents(chunks)
        for i, chunk in enumerate(chunks):
            insert_document(chunk, embeddings[i])

def process_uploaded_files(uploaded_files, chunk_size, overlap, embedding_model_name, progress_bar, progress_text):
    create_table()
    total_files = len(uploaded_files)
    embedding_model = get_embedding_model(embedding_model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, uploaded_file in enumerate(uploaded_files):
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
                        _process_file(os.path.join(root, file), embedding_model, chunk_size, overlap)
            else:
                _process_file(file_path, embedding_model, chunk_size, overlap)
                
            progress_bar.progress((i + 1) / total_files)
            progress_text.text(f"Processando arquivo {i + 1}/{total_files}: {uploaded_file.name}")
    progress_text.text("Processamento concluído.")