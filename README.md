# Dr. X NLP Project
Analyzes Dr. X's publications using FAISS and Ollama.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install Ollama and pull models: `ollama pull llama3 nomic-embed-text`
3. Place files in `data/`.
4. Run: `python main.py`

## Methodology
- Text extraction with `python-docx`, `PyPDF2`, `pandas`.
- Chunking with `tiktoken` (cl100k_base).
- Vector DB with FAISS and nomic embeddings.
- RAG with Ollama’s Llama 3.
- Translation and summarization with Llama 3.
- Performance measured in tokens/second.

## Models
- LLM: Llama 3 (via Ollama)
- Embeddings: nomic-embed-text (via Ollama)