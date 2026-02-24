from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Embedding Model (for initial retrieval)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Re-ranking Model (for accuracy)
rerank_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Converts a list of text chunks into embedding vectors.
    """
    return embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string.
    """
    return embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)

def rerank_chunks(query: str, chunks: list[str], top_n: int) -> list[str]:
    """
    Scores chunks against the query and returns the top_n most relevant ones.
    """
    if not chunks:
        return []
    
    # Pairs for cross-encoding: [[query, chunk1], [query, chunk2], ...]
    pairs = [[query, chunk] for chunk in chunks]
    scores = rerank_model.predict(pairs)
    
    # Sort chunks by score in descending order
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    
    return ranked_chunks[:top_n]