import faiss
import json
import numpy as np
import os
from pathlib import Path
from ollama import Client

def create_vector_db(chunks, model_name="nomic-embed-text"):
    """Create FAISS vector database from chunks."""
    ollama = Client()
    dimension = 768  # Nomic embed dimension
    index = faiss.IndexFlatL2(dimension)
    embeddings = []
    metadata = []

    for chunk in chunks:
        response = ollama.embeddings(model=model_name, prompt=chunk["text"])
        embedding = np.array(response["embedding"], dtype=np.float32)
        embeddings.append(embedding)
        metadata.append(chunk)

    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    # Save index and metadata
    os.makedirs("outputs", exist_ok=True)
    faiss.write_index(index, os.path.join("..","outputs", "vector_db.faiss"))
    with open(os.path.join("..","outputs", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return index, metadata

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    chunks_dir = os.path.join(base_dir,"outputs", "chunks")

    all_chunks = []
    if not os.path.exists(chunks_dir):
        print(f"Directory '{chunks_dir}' does not exist.")
    else:
        for root, _, files in os.walk(chunks_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    print(f"Loading chunks from: {file_path}")
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                        all_chunks.extend(chunks)

    if all_chunks:
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        create_vector_db(all_chunks)
        print("Vector database created successfully.")
    else:
        print("No chunks found.")
