import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def create_vector_db(chunks, model_name="nomic-embed-text"):
    """Create FAISS vector database from chunks using LangChain's FAISS.from_documents."""
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=model_name)

        # Convert chunks to LangChain Document objects
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "file_name": chunk["file_name"],
                    "chunk_number": chunk["chunk_number"]
                }
            )
            for chunk in chunks
        ]

        # Create FAISS vector store from documents
        vector_db = FAISS.from_documents(documents, embeddings)

        # Save index
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        vector_db.save_local(output_dir / "vector_db")

        return vector_db

    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None

def add_chunk_to_vector_db(chunk, model_name="nomic-embed-text"):
    """Add a single chunk to an existing FAISS vector database."""
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=model_name)

        # Load existing FAISS vector store
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "outputs"
        vector_db_path = output_dir / "vector_db"

        if not vector_db_path.exists():
            print(f"Vector database at '{vector_db_path}' does not exist.")
            return False

        vector_db = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Convert chunk to LangChain Document
        document = Document(
            page_content=chunk["text"],
            metadata={
                "file_name": chunk["file_name"],
                "chunk_number": chunk["chunk_number"]
            }
        )

        # Add document to vector store
        vector_db.add_documents([document])

        # Save updated vector store
        vector_db.save_local(vector_db_path)

        print(f"Successfully added chunk {chunk['chunk_number']} from {chunk['file_name']} to vector database.")
        return True

    except Exception as e:
        print(f"Error adding chunk to vector database: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    chunks_dir = base_dir / "outputs" / "chunks"

    # Load all chunks for initial creation
    all_chunks = []
    if not chunks_dir.exists():
        print(f"Directory '{chunks_dir}' does not exist.")
    else:
        for file_path in chunks_dir.rglob("*.json"):
            print(f"Loading chunks from: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if all_chunks:
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        vector_db = create_vector_db(all_chunks)
        if vector_db:
            print("Vector database created successfully.")
        else:
            print("Failed to create vector database.")
    else:
        print("No chunks found.")

    # Example: Add a new chunk
    new_chunk = {
        "file_name": "example.txt",
        "chunk_number": 999,
        "text": "This is a new chunk added to the vector database."
    }
    print("\nAdding a new chunk to the vector database...")
    success = add_chunk_to_vector_db(new_chunk)
    if not success:
        print("Failed to add new chunk.")