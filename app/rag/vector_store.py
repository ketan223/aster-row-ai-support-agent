import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

_model_instance = None

# Returns a singleton instance of the local SentenceTransformer model.
def get_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        # Using a practical, free local sentence-transformer model.
        _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_instance

class LocalVectorStore:
    # Initializes the local vector store, optionally loading from a pre-built index file.
    def __init__(self, index_path: str = None):
        self.chunks = []
        self.embeddings = None
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    # Loads the indexed chunks and precomputed embeddings from a JSON file.
    def load_index(self, index_path: str):
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        raw_embeddings = data.get("embeddings", [])
        if raw_embeddings:
            self.embeddings = np.array(raw_embeddings, dtype=np.float32)
        else:
            self.embeddings = None

    # Saves the currently loaded chunks and computed embeddings to a JSON file.
    def save_index(self, index_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        raw_embeddings = self.embeddings.tolist() if self.embeddings is not None else []
        data = {
            "chunks": self.chunks,
            "embeddings": raw_embeddings
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Computes embeddings for the provided list of chunks and stores them.
    def build_from_chunks(self, chunks: list[dict]):
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return
            
        model = get_embedding_model()
        texts = [c["text"] for c in chunks]
        # Encode returns a numpy array
        embs = model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(embs, dtype=np.float32)

    # Searches the vector store for the top_k most similar chunks using cosine similarity.
    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if not self.chunks or self.embeddings is None:
            return []
            
        model = get_embedding_model()
        query_emb = model.encode(query, show_progress_bar=False)
        query_vector = np.array(query_emb, dtype=np.float32)
        
        # Calculate cosine similarity using NumPy
        dot_products = np.dot(self.embeddings, query_vector)
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Avoid zero division
        norms = query_norm * doc_norms
        norms = np.where(norms == 0, 1e-9, norms)
        
        similarities = dot_products / norms
        
        # Get sorted indices in descending order of similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append((self.chunks[idx], float(similarities[idx])))
            
        return results
