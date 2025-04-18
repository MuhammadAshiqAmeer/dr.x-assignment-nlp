
# 🧠 AI Engineer - NLP Skills Assessment

This project is a modular and extensible **Natural Language Processing (NLP) pipeline** that can:
- Extract text from `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx` files
- Chunk the text using `langchain` split_text
- Embed using `nomic-embed-text` via Ollama
- Store embeddings in a FAISS vector store
- Enable Retrieval-Augmented Generation (RAG) querying using `llama3`
- Translate and summarize documents
- Log performance tokens/second for profiling

---

## 🔧 Setup Instructions

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and configure Ollama
```bash
ollama pull llama3
ollama pull nomic-embed-text
```
Make sure Ollama is running locally (`ollama serve` if necessary).

### 3. Prepare data
Put all your input files (`.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`) inside the a directory.

---

## 🚀 Usage Examples

### Run the full pipeline (extract, chunk, embed, and enable RAG)
```bash
python main.py --data-dir "data-directory-path" --rag
```

### Translate a single document
```bash
python main.py --input-file "data/sample.pdf" --translate --target-lang ar
```

### Summarize a single document
```bash
python main.py --input-file "data/sample.pdf" --summarize --summary-strategy abstractive
```

### Translate & Summarize a document in one go
```bash
python main.py --input-file "data/sample.pdf" --translate --summarize --target-lang en --summary-strategy extractive
```

### Start RAG-only chat (FAISS DB must already exist)
```bash
python main.py --rag
```

### Add extra data to existing vector db
```bash
python main.py --add-data "file-path"
```

---

## ⚙️ CLI Parameters

| Argument              | Description                                                     |
|-----------------------|-----------------------------------------------------------------|
| `--data-dir`          | Directory containing input files (default: `data/`)             |
| `--input-file`        | Process a single file (translate or summarize)                  |
| `--rag`               | Start an interactive RAG chatbot session                        |
| `--add-data`          | Add extra data to knowledge base                                |
| `--translate`         | Enable translation                                              |
| `--summarize`         | Enable summarization                                            |
| `--target-lang`       | Language to translate into (default: `en`, options: `en`, `ar`) |
| `--summary-strategy`  | `abstractive` (default) or `extractive` summarization           |
| `--max-chars`         | Limit characters per file for summarization/translation         |

---

## 🔍 Pipeline Methodology

- **Text Extraction**: 
  - Used langchain document loaders. 
  - `.docx`: `python-docx` ,`UnstructuredWordDocumentLoader` 
  - `.pdf`: `PyMuPDFLoader` 
  - `.csv`: `CSVLoader` 
  - `xlxs`,`xls` ,`xlxm` : `UnstructuredExcelLoader`
  - `.txt`: `TextLoader` 

- **Chunking**:  
  - Uses `langchain` text splitter `RecursiveCharacterTextSplitter`
  - Default chunk size: 1000 tokens  
  - Overlap: 100 tokens for context preservation  

- **Embedding**:  
  - Uses `nomic-embed-text` via Ollama API  
  - Stores vectors in FAISS (L2 norm)  
  - Metadata saved in JSON alongside `.faiss` index  

- **RAG (Retrieval-Augmented Generation)**:  
  - Top-k chunk retrieval  
  - Merged context + user query  
  - Passed to `llama3` via Ollama to generate response  

- **Translation**:  
  - Automatically detects source language using `langdetect`  
  - Translates to `target_lang`  
  - Fluency improved using post-pass via `llama3`  

- **Summarization**:  
  - Recursive chunked summarization for long documents  
  - `abstractive` or `extractive` supported  
  - ROUGE evaluation against reference  

- **Performance Logging**:  
  - Logs token throughput (tokens/sec) for each task  
  - Stored in `outputs/performance.json`  

---

## 📁 Output Folder Structure

```
outputs/
├── chunks/                # JSON files of chunked document content
├── summaries/             # Summarized output for each file
├── translated/            # Translated documents
├── metadata.json          # Metadata used in FAISS vector DB
├── vector_db              # FAISS index storing embedded vectors and documents
├── performance.json       # Token throughput performance logs
└── pipeline.log           # Runtime logs for debugging
```

---

## 🧠 Models Used

| Task         | Model via Ollama                                                |
|--------------|-----------------------------------------------------------------|
| Embedding    | `nomic-embed-text`                                              |
| Generation   | `llama3:8b` (or other variants like `llama3:7b`, `gemma`, etc.) |

---

## 📊 Performance

Each NLP task (translation, chunking, embedding, summarizing) logs its processing speed in terms of tokens per second. This helps optimize model runtime and track latency over time.

---

## 💡To improve

- The .docx files contains tables. Tables can be extracted using docx library. Extracted table need to be splitted using langchains `MarkdownHeaderTextSplitter` for better results. The remaining paragraph texts using `RecursiveCharacterTextSplitter`. So chunking the table seperatly disconnects both. Need to integrate an optimal solution.

## 🪪 License

MIT License – open-source and free for educational, personal, and research use.

---

## 🙋‍♂️ Author

Created by **Muhammad Ashiq Ameer**  
Built using:
- 🦜 LangChain  
- 🧠 Ollama  
- 🧲 FAISS  
