import os
import yaml
import datetime

# Recursively converts date and datetime objects in a nested structure 
# to ISO 8601 string format to prevent JSON serialization errors.
def _stringify_dates(val):
    if isinstance(val, dict):
        return {k: _stringify_dates(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_stringify_dates(v) for v in val]
    elif isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    return val

# Parses a single markdown file, extracting YAML front-matter metadata 
# and separation of front matter from the actual content.
def load_markdown_document(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    metadata = {}
    content = raw_text
    
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            yaml_text = parts[1]
            content = parts[2].strip()
            try:
                metadata = yaml.safe_load(yaml_text) or {}
                metadata = _stringify_dates(metadata)
            except Exception as e:
                # Fallback to empty metadata on parse error
                metadata = {}
                
    metadata["filename"] = os.path.basename(file_path)
    return {
        "metadata": metadata,
        "content": content,
        "filename": os.path.basename(file_path)
    }

# Scans a directory for markdown files and loads them into a list of document dicts.
def load_all_documents(dir_path: str) -> list[dict]:
    documents = []
    if not os.path.exists(dir_path):
        return documents
        
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".md"):
            file_path = os.path.join(dir_path, filename)
            doc = load_markdown_document(file_path)
            documents.append(doc)
            
    return documents
