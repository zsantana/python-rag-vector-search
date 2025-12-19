from app.domain.search import VectorSearch, SemanticSearch, HybridSearch

def get_search_strategy(search_type, embedding_model_name, vector_weight=None, minimum_score=None):
    if search_type == "Vetorial":
        return VectorSearch(embedding_model_name)
    elif search_type == "Semântica":
        return SemanticSearch()
    elif search_type == "Híbrida":
        return HybridSearch(embedding_model_name, vector_weight, minimum_score)
    else:
        raise ValueError("Tipo de busca inválido")

def search(query, search_strategy, top_k):
    return search_strategy.search(query, top_k)
