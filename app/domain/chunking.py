from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return text_splitter.split_text(text)
