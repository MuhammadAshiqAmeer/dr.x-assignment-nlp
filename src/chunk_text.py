import tiktoken
import json
import os
from pathlib import Path

def chunk_text(text, file_name, max_tokens=400, overlap_tokens=50):
    """Chunk text using cl100k_base tokenizer with overlap and better page estimation."""
    if not text.strip():
        return []  # Handle empty text

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    chunks = []
    chunk_num = 1
    text_length = len(text)
    avg_chars_per_page = 2000  # Estimate: adjust based on your documents

    i = 0
    while i < len(tokens):
        # Define chunk end, respecting max_tokens
        end_idx = min(i + max_tokens, len(tokens))
        chunk_tokens = tokens[i:end_idx]
        chunk_text = encoding.decode(chunk_tokens)

        # Estimate page number based on character position
        char_pos = len(encoding.decode(tokens[:i]))
        page_num = max(1, (char_pos // avg_chars_per_page) + 1)

        # Create chunk metadata
        chunks.append({
            "file_name": os.path.basename(file_name),  # Store only file name
            "page_number": page_num,
            "chunk_number": chunk_num,
            "text": chunk_text.strip()  # Remove leading/trailing whitespace
        })
        chunk_num += 1

        # Move to next chunk with overlap
        i += max_tokens - overlap_tokens
        if i >= len(tokens):
            break

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
    # Use absolute paths for robustness
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