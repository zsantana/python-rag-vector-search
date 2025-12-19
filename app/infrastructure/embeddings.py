from langchain_openai import OpenAIEmbeddings

def get_embedding_model(model_name):
    return OpenAIEmbeddings(model=model_name)
