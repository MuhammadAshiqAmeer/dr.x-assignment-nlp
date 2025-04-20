# 🧠 AI Engineer – NLP Pipeline & RAG Skills Assessment

A **modular, extensible NLP pipeline** that handles diverse document types (books, research papers, reports, PDFs, DOCX, CSV, Excel, TXT), and supports:

- **Text Extraction** from `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`  
- **Table Extraction & Chunking** from Word documents  
- **Chunking** with smart overlaps & semantic separators  
- **Embedding** via `nomic-embed-text` (Ollama) → FAISS vector store  
- **Retrieval‑Augmented Generation (RAG)** using `llama3:8b`  
- **Translation** with automatic language detection, single‑pass fluency refinement  
- **Summarization** (abstractive or extractive) with structured prompts, recursive reduction  
- **Progress visualization** on console with `tqdm`  
- **Performance logging** (tokens/sec) for each stage  

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
Put all your input files (`.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`) inside a directory.

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

## ⚙️ CLI Arguments

| Flag                   | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `--data-dir`           | Directory of input files (default `data/`)                             |
| `--input-file`         | Single file to translate/summarize                                     |
| `--rag`                | Launch interactive RAG chat                                            |
| `--add-data`           | Add files to existing FAISS vector store                               |
| `--translate`          | Perform translation                                                    |
| `--target-lang`        | Translation target (default `en`)                                      |
| `--summarize`          | Perform summarization                                                  |
| `--summary-strategy`   | `abstractive` (default) or `extractive`                                |
| `--max-chars`          | Max characters per chunk (default full text)                           |

---

## 🔍 Pipeline Overview

1. **Text Extraction**  
   - PDF → `PyMuPDFLoader`  
   - DOCX → `python-docx` + table extractor  
   - CSV → `CSVLoader`  
   - Excel → `UnstructuredExcelLoader`  
   - TXT → `TextLoader`

2. **Table Extraction**  
   - Detect `.docx` tables  
   - Chunk by rows (`rows_per_chunk`)  
   - Export each as JSON with headers & rows

3. **Chunking**  
   - `RecursiveCharacterTextSplitter`  
   - `chunk_size` (chars) + `chunk_overlap`  
   - Splits at `\n`, ` `, `. `, etc.

4. **Embedding**  
   - `nomic-embed-text` via Ollama  
   - Store in FAISS (L2 norm) + JSON metadata

5. **RAG**  
   - Top‑k retrieval of chunks  
   - Concatenate with user query  
   - Generate answer via `llama3:8b`

6. **Translation**  
   - Detect source with `langdetect`  
   - Single‑pass prompt: translate + refine fluency  

7. **Summarization**  
   - Chapter‑ or character‑based splitting when possible  
   - Structured prompts (plot points, characters, themes)  
   - Recursive summarization until target size  
   - Support for both abstractive & extractive  

8. **Progress & Performance**  
   - `tqdm` progress bars for chunk processing  
   - Log tokens/sec in `outputs/performance.json`  

---

## 📁 Outputs

```
outputs/
├── chunks/             # JSON chunks (text & tables)
├── summaries/          # Summaries per file
├── translated/         # Translated outputs
├── metadata.json       # FAISS metadata
├── vector_db/          # FAISS index files
├── performance.json    # Token throughput logs
└── pipeline.log        # Detailed runtime logs
```

---

## 🧠 Models

| Task        | Model (via Ollama)              |
|-------------|---------------------------------|
| Embedding   | `nomic-embed-text`              |
| Generation  | `llama3:8b` (variants available)|

---

## 🪪 License

MIT — free for educational, personal, and research use.

---

## 🙋‍♂️ Author

**Muhammad Ashiq Ameer**  
Built with LangChain · Ollama · FAISS  
