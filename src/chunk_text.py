import tiktoken
import json
import os

def chunk_text(text, file_name, max_tokens=512):
    """Chunk text using cl100k_base tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    chunk_num = 1
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        # Estimate page number (simplified)
        page_num = (i // max_tokens) + 1
        chunks.append({
            "file_name": file_name,
            "page_number": page_num,
            "chunk_number": chunk_num,
            "text": chunk_text
        })
        chunk_num += 1
    
    # Ensure chunks_dir exists
    os.makedirs(chunks_dir, exist_ok=True)
    # Properly construct the output path
    output_filename = os.path.splitext(os.path.basename(file_name))[0] + ".json"
    output_path = os.path.join(chunks_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=4)
    return chunks


if __name__ == "__main__":
    extracted_dir = os.path.join("..","outputs","extracted")
    chunks_dir = os.path.join("..","outputs","chunks")
    
    os.makedirs(chunks_dir, exist_ok=True)
    
    if not os.path.exists(extracted_dir):
        print(f"Directory '{extracted_dir}' does not exist.")
    else:
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Chunking: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chunks = chunk_text(text, file_path)
                print(f"Chunked into {len(chunks)} pieces: {file}")
