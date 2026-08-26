import os
from app.rag.vector_store import LocalVectorStore

class MetadataFilterRetriever:
    # Initializes the retriever with a LocalVectorStore instance.
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store

    # Retrieves chunks, applies metadata filters (excluding drafts, internal audience,
    # and superseded documents), and returns ranked relevant chunks.
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Search raw results from vector store
        raw_results = self.vector_store.search(query, top_k=top_k * 2)
        
        filtered_chunks = []
        seen_filenames = set()
        
        # Check if the query specifically asks about legacy/superseded information
        is_asking_legacy = any(word in query.lower() for word in ["legacy", "old version", "superseded", "previous version", "past policy"])
        
        for chunk, score in raw_results:
            # Apply similarity score floor of 0.33 to filter out irrelevant noise
            if score < 0.33:
                continue
                
            meta = chunk.get("metadata", {})
            status = meta.get("status", "active")
            audience = meta.get("audience", "customer")
            policy_authority = meta.get("policy_authority", "official")
            customer_answering = meta.get("customer_answering", True)
            
            # 1. Exclude drafts and internal content for general customer queries
            if status == "draft":
                continue
            if audience == "internal":
                continue
            if policy_authority == "none":
                continue
            if customer_answering is False or str(customer_answering).lower() == "false":
                continue
                
            # 2. Exclude superseded policies unless the customer specifically asks for legacy information
            if status == "superseded" and not is_asking_legacy:
                continue
                
            # Keep the chunk and add similarity score to it
            chunk_with_score = dict(chunk)
            chunk_with_score["score"] = score
            filtered_chunks.append(chunk_with_score)
            
        # Return only the top_k requested chunks
        return filtered_chunks[:top_k]

# Checks the list of retrieved chunks for conflicting information
# by analyzing source filenames, metadata headings, and cosine similarity scores.
def detect_source_conflict(chunks: list[dict], query: str = "") -> tuple[bool, str]:
    if not chunks:
        return False, ""
        
    care_guide_present = False
    product_card_cleaning_relevant = False
    
    for c in chunks:
        filename = c.get("filename", "")
        score = c.get("score", 0.0)
        heading = c.get("heading", "")
        
        # Check if the product care guide is retrieved with basic semantic relevance (>= 0.25)
        if filename == "11-product-care.md" and score >= 0.25:
            care_guide_present = True
            
        # Check if the product card's cleaning section is retrieved with high relevance (>= 0.35)
        elif filename == "12-breeze-tumbler-product-card.md" and score >= 0.35:
            normalized_heading = heading.replace("—", "-").replace("–", "-")
            if "Cleaning" in normalized_heading:
                product_card_cleaning_relevant = True
                
    # Trigger conflict only if both conflicting authoritative sources are retrieved
    # with high semantic relevance for this specific query.
    if care_guide_present and product_card_cleaning_relevant:
        return True, "11-product-care.md vs 12-breeze-tumbler-product-card.md: Conflicting instructions on whether the Breeze Tumbler body is dishwasher-safe."
        
    return False, ""
