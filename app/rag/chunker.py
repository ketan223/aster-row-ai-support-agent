import re

# Splits a document's content by markdown headings (#, ##, ###) 
# and returns a list of chunk dictionaries with inherited metadata.
def chunk_document(doc: dict) -> list[dict]:
    content = doc["content"]
    metadata = doc["metadata"]
    filename = doc["filename"]
    
    doc_title = metadata.get("title", filename)
    
    lines = content.split("\n")
    chunks = []
    
    current_heading = "Introduction"
    current_lines = []
    
    h1 = doc_title
    h2 = ""
    h3 = ""
    
    # Helper to flush the accumulated lines into a chunk
    def flush_chunk():
        nonlocal current_lines, current_heading, h1, h2, h3
        text = "\n".join(current_lines).strip()
        if text:
            # Reconstruct full heading path for clarity in citation if needed
            heading_path = current_heading
            chunks.append({
                "text": text,
                "heading": heading_path,
                "document_title": doc_title,
                "filename": filename,
                "metadata": metadata
            })
        current_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Check for headings
        h1_match = re.match(r"^#\s+(.+)$", stripped)
        h2_match = re.match(r"^##\s+(.+)$", stripped)
        h3_match = re.match(r"^###\s+(.+)$", stripped)
        
        if h1_match:
            flush_chunk()
            h1 = h1_match.group(1).strip()
            h2 = ""
            h3 = ""
            current_heading = h1
            current_lines.append(line)
        elif h2_match:
            flush_chunk()
            h2 = h2_match.group(1).strip()
            h3 = ""
            current_heading = h2
            current_lines.append(line)
        elif h3_match:
            flush_chunk()
            h3 = h3_match.group(1).strip()
            current_heading = h3
            current_lines.append(line)
        else:
            current_lines.append(line)
            
    flush_chunk()
    return chunks

# Chunks multiple loaded documents, returning a flat list of all chunks.
def chunk_all_documents(documents: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
