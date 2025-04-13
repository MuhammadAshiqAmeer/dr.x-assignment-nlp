# AI Engineer - NLP Skills Assessment

A modular NLP pipeline that extracts, chunks, embeds, and queries scientific documents using FAISS and LLMs via Ollama.

---

## 🔧 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and configure Ollama
```bash
ollama pull llama3
ollama pull nomic-embed-text
```
Ensure Ollama is running locally.

### 3. Prepare input data
Place your `.pdf`, `.docx`, `.txt`, or spreadsheet files into the `data/` folder.

---

## 🚀 Usage

### Full pipeline (extract, chunk, embed, and query):
```bash
python main.py --data-dir data --rag
```

### Translate and summarize a single file:
```bash
python main.py --input-file "data/sample.pdf" --translate --target-lang "ar"
```

### Summarize a single file:
```bash
python main.py --input-file "data/sample.pdf" --summarize --summary-strategy abstractive
```

### Translate and summarize a single file:
```bash
python main.py --input-file data/sample.pdf --translate --summarize --target-lang en --summary-strategy abstractive
```

### Start only the RAG chat (must have vector DB ready):
```bash
python main.py --rag
```

---

## ⚙️ CLI Options
| Option              | Description |
|---------------------|-------------|
| `--data-dir`        | Directory containing documents to process (default: `data`) |
| `--input-file`      | Process a single file (translate/summarize only) |
| `--rag`             | Start interactive RAG session |
| `--translate`       | Translate the text before further processing |
| `--summarize`       | Summarize the text (requires `--input-file`) |
| `--target-lang`     | Translation target language (default: `en`) |
| `--summary-strategy`| Type of summarization (`abstractive` or `extractive`) |
| `--max-chars`       | Max characters to process from input (default: 5000) |

---

## 🔍 Methodology

- **Text Extraction**: Handles `.pdf`, `.docx`, `.csv`, `.xlsx` using `PyMuPDF`, `python-docx`, and `pandas`.
- **Chunking**: Uses `tiktoken` with `cl100k_base` tokenizer, chunk size 400, overlap 50.
- **Embedding**: Uses `nomic-embed-text` (via Ollama) and FAISS index for fast similarity search.
- **RAG**: Combines top-k relevant chunks with user question, sent to `llama3` for context-based response.
- **Translation & Summarization**: Uses `llama3` to translate and summarize user content.
- **Performance Logging**: Stores token throughput in `outputs/performance.json`.

---

## 📁 Output Structure
```
outputs/
├── chunks/                # Stored JSON chunks
├── summaries/             # Summarized results
├── translated/            # Translated text files
├── metadata.json          # Metadata for vector DB
├── vector_db.faiss        # FAISS vector index
├── performance.json       # Performance logs
└── pipeline.log           # Runtime logs
```

---

## 🧠 Models Used
- **LLM**: `llama3` (Ollama)
- **Embeddings**: `nomic-embed-text` (Ollama)

---

## 📜 License
MIT License

---

## 🙋‍♂️ Author
Built by Muhammad Ashiq Ameer – powered by LangChain, Ollama, and FAISS.

---

