import json
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, file_name, max_tokens=1000, overlap_tokens=100):
    """Chunk text using LangChain's RecursiveCharacterTextSplitter."""
    if not text.strip():
        return []

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )

    chunks = []
    chunk_num = 1
    avg_chars_per_page = 2000  # Estimate: adjust based on your documents

    # Create documents to get start_index metadata
    documents = text_splitter.create_documents([text])
    
    for doc in documents:
        # Get start_index from document metadata
        char_pos = doc.metadata.get("start_index", 0)
        page_num = max(1, (char_pos // avg_chars_per_page) + 1)

        # Create chunk metadata
        chunks.append({
            "file_name": os.path.basename(file_name),
            "page_number": page_num,
            "chunk_number": chunk_num,
            "text": doc.page_content.strip()
        })
        chunk_num += 1

    return chunks

def save_chunks(chunks, file_name, chunks_dir):
    """Save chunks to JSON file."""
    if not chunks:
        return
    os.makedirs(chunks_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(file_name))[0] + ".json"
    output_path = os.path.join(chunks_dir, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    extracted_dir = base_dir / "outputs" / "extracted"
    chunks_dir = base_dir / "outputs" / "chunks"

    try:
        if not extracted_dir.exists():
            print(f"Directory '{extracted_dir}' does not exist.")
        else:
            for file_path in extracted_dir.rglob("*.*"):
                if file_path.is_file():
                    print(f"Chunking: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        chunks = chunk_text(text, str(file_path), max_tokens=400, overlap_tokens=50)
                        save_chunks(chunks, str(file_path), chunks_dir)
                        print(f"Chunked into {len(chunks)} pieces: {file_path.name}")
                    except UnicodeDecodeError:
                        print(f"Encoding error in {file_path}; skipping.")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    except Exception as e:
        print(f"Error accessing directory {extracted_dir}: {e}")