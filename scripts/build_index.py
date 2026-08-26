import os
import sys

# Ensure project root is in the Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.document_loader import load_all_documents
from app.rag.chunker import chunk_all_documents
from app.rag.vector_store import LocalVectorStore

# Builds the local vector index by loading all markdown docs, chunking them,
# generating embeddings, and writing to the target JSON path.
def build_index():
    print("Loading documents from knowledge base...")
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge-base")
    documents = load_all_documents(kb_dir)
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting documents into chunks...")
    chunks = chunk_all_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    
    print("Computing embeddings and building index...")
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indexes", "kb_index.json")
    store = LocalVectorStore()
    store.build_from_chunks(chunks)
    
    print(f"Saving vector index to {index_path}...")
    store.save_index(index_path)
    print("Index successfully built!")

if __name__ == "__main__":
    build_index()
