import os
import argparse
import logging
import time
import json
from pathlib import Path
from typing import List, Tuple
from src.extract_text import extract_text_from_file
from src.chunk_text import chunk_text, save_chunks
from src.vector_db import create_vector_db
from src.rag import RAGSystem
from src.translate import translate_text
from src.summarize import summarize_text, evaluate_summary
from src.utils import measure_performance, save_text
from src.extract_table_and_chunk_docx import process_file 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

log_path = "outputs/pipeline.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



def init_performance_log(output_path: str = "outputs/performance.json") -> None:
    """Initialize performance.json as an empty list if it doesn't exist."""
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)



def log_performance(task_name: str, tokens_per_second: float, output_path: str = "outputs/performance.json") -> None:
    """Append performance metrics to performance.json."""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append({"task": task_name, "tokens_per_second": tokens_per_second, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)



def extract_and_chunk(data_dir: str, chunks_dir: str = "outputs/chunks") -> List[dict]:
    """Extract text from files and chunk them, measuring performance."""
    all_chunks = []
    for file_path in Path(data_dir).rglob("*.*"):
        if not file_path.is_file():
            continue
        logger.info(f"Processing file: {file_path}")
        try:
            extension = file_path.suffix.lower()
            if extension == '.docx':
                # Handle .docx files with table extraction
                chunks = measure_performance(
                    str(file_path),
                    lambda _: process_file(str(file_path), chunks_dir),
                    f"extract_and_chunk_docx_{file_path.name}"
                )
            else:
                # Handle other file types with text extraction
                text = measure_performance(
                    str(file_path),
                    lambda _: extract_text_from_file(str(file_path)),
                    f"extract_{file_path.name}"
                )
                if not text or not text.strip():
                    logger.warning(f"Empty or failed extraction: {file_path}")
                    continue
                chunks = measure_performance(
                    text,
                    lambda t: chunk_text(t, str(file_path), max_tokens=1500, overlap_tokens=100),
                    f"chunk_{file_path.name}"
                )
                save_chunks(chunks, str(file_path), chunks_dir)
            
            if chunks:
                all_chunks.extend(chunks)
                logger.info(f"Extracted and chunked {len(chunks)} chunks from {file_path}")
            else:
                logger.warning(f"No chunks created for {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    return all_chunks



def build_vector_db(chunks: List[dict], output_dir: str = "outputs") -> None:
    """Build vector database if it doesn't exist, measuring performance."""
    vector_db_path = Path(output_dir) / "vector_db"
    if vector_db_path.exists():
        logger.info("Vector database already exists. Skipping creation.")
        return
    logger.info("Creating vector database...")
    measure_performance(
        "".join(chunk["text"] for chunk in chunks),
        lambda _: create_vector_db(chunks),
        "vectordb_creation"
    )
    logger.info("Vector database created.")



def add_single_document(file_path: str, vector_db_path: str = "outputs/vector_db", chunks_dir: str = "outputs/chunks") -> None:
    """Add a single document to the existing FAISS vector store."""
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"File {file_path} does not exist or is not a file.")
        return
    if not Path(vector_db_path).exists():
        logger.error(f"Vector database {vector_db_path} does not exist. Run pipeline with --data-dir first.")
        return

    logger.info(f"Adding single document: {file_path}")
    try:
        extension = file_path.suffix.lower()
        if extension == '.docx':
            # Handle .docx files with table extraction
            chunks = measure_performance(
                str(file_path),
                lambda _: process_file(str(file_path), chunks_dir),
                f"extract_and_chunk_docx_{file_path.name}"
            )
        else:
            # Handle other file types with text extraction
            text = measure_performance(
                str(file_path),
                lambda _: extract_text_from_file(str(file_path)),
                f"extract_{file_path.name}"
            )
            if not text or not text.strip():
                logger.error(f"Empty or failed extraction: {file_path}")
                return
            chunks = measure_performance(
                text,
                lambda t: chunk_text(t, str(file_path), max_tokens=1500, overlap_tokens=100),
                f"chunk_{file_path.name}"
            )
            save_chunks(chunks, str(file_path), chunks_dir)

        if not chunks:
            logger.error(f"No chunks created for {file_path}")
            return

        logger.info(f"Extracted and chunked {len(chunks)} chunks from {file_path}")

        # Load existing FAISS vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

        # Convert chunks to LangChain Documents
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "file_name": chunk["file_name"],
                    "page_number": chunk.get("page_number", 0),
                    "chunk_number": chunk["chunk_number"],
                    "section_type": chunk.get("section_type", "text")
                }
            ) for chunk in chunks
        ]

        # Add documents to vector store
        measure_performance(
            "".join(chunk["text"] for chunk in chunks),
            lambda _: vector_store.add_documents(documents),
            f"add_document_{file_path.name}"
        )

        # Save updated vector store
        vector_store.save_local(vector_db_path)
        logger.info(f"Added {len(chunks)} chunks from {file_path} to vector database.")

    except Exception as e:
        logger.error(f"Error adding {file_path} to vector database: {e}")



def run_rag_interactive(vector_db_path: str) -> None:
    """Start an interactive RAG session."""
    logger.info("Starting interactive RAG session. Type 'exit' to quit.")
    try:
        rag = RAGSystem(vector_db_path)
        while True:
            question = input("🧠 You: ")
            if question.strip().lower() in ["exit", "quit"]:
                logger.info("Exiting RAG session.")
                break
            if not question.strip():
                logger.warning("Empty question. Please enter a valid question.")
                continue
            answer = measure_performance(question, rag.query, f"rag_{question[:20]}")
            print(f"\n🤖 Assistant: {answer}\n")
    except Exception as e:
        logger.error(f"RAG session failed: {e}")



def process_text(text: str, file_name: str, target_lang: str = None, summary_strategy: str = None) -> Tuple[str, str, dict]:
    """Translate and/or summarize text, measuring performance."""
    translated = text
    summary = ""
    scores = {}

    if target_lang:
        logger.info(f"Translating {file_name} to {target_lang}...")
        translated = measure_performance(
            text,
            lambda t: translate_text(t, target_lang=target_lang),
            f"translate_{file_name}"
        )
        # logger.info(f"Translated (first 100 chars): {translated[:100]}...")

    if summary_strategy:
        text_to_summarize = translated if target_lang else text
        logger.info(f"Summarizing {file_name} ({summary_strategy})...")
        summary = measure_performance(
            text_to_summarize,
            lambda t: summarize_text(t, strategy=summary_strategy),
            f"summarize_{file_name}"
        )
        # logger.info(f"Summary (first 100 chars): {summary[:100]}...")
        # logger.info("Evaluating summary...")
        # scores = evaluate_summary(text_to_summarize[:1000], summary)
        # logger.info("ROUGE Scores:")
        # for k, v in scores.items():
        #     logger.info(f"{k.upper()}: P={v.precision:.2f}, R={v.recall:.2f}, F1={v.fmeasure:.2f}")

    return translated, summary, scores



def main(args: argparse.Namespace) -> None:
    """Main pipeline orchestrator."""
    start_time = time.time()
    logger.info("Starting NLP pipeline...")
    init_performance_log()  # Initialize performance.json

    # Ensure output directories exist
    Path("outputs/chunks").mkdir(parents=True, exist_ok=True)
    Path("outputs/translated").mkdir(parents=True, exist_ok=True)
    Path("outputs/summaries").mkdir(parents=True, exist_ok=True)

    # Handle adding a single document to the vector store
    if args.add_data:
        add_single_document(args.add_data)
        logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
        return

    # Handle single file translation/summarization
    if args.input_file:
        file_path = Path(args.input_file)
        if not file_path.is_file():
            logger.error(f"Input file {file_path} does not exist or is not a file.")
            return
        logger.info(f"Processing single file: {file_path}")
        try:
            text = extract_text_from_file(str(file_path))
            if not text or not text.strip():
                logger.error(f"Empty or failed extraction: {file_path}")
                return
            translated, summary, scores = process_text(
                text[:args.max_chars],
                file_path.name,
                args.target_lang if args.translate else None,
                args.summary_strategy if args.summarize else None
            )
            if args.translate:
                output_path = f"outputs/translated/{file_path.stem}_{args.target_lang}.txt"
                save_text(translated, output_path)
                logger.info(f"Saved translation to {output_path}")
            if args.summarize:
                output_path = f"outputs/summaries/{file_path.stem}_{args.summary_strategy}.txt"
                save_text(summary, output_path)
                logger.info(f"Saved summary to {output_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
        return

    if args.rag:
        vector_db_path = Path("outputs") / "vector_db"
        if args.data_dir:
            # Handle full pipeline (data_dir and RAG)
            all_chunks = extract_and_chunk(args.data_dir)
            if not all_chunks:
                logger.error("No chunks created, check the path. Aborting pipeline.")
                return

            if all_chunks:
                build_vector_db(all_chunks)

        if not vector_db_path.exists():
            logger.error("Vector database not found. Run pipeline with data_dir first.")
            return
        
        run_rag_interactive(str(vector_db_path))

    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Pipeline for Dr. X's Publications")
    parser.add_argument("--data-dir", help="Directory containing input files")
    parser.add_argument("--input-file", help="Single text file to translate or summarize")
    parser.add_argument("--add-data", help="Single file to add to the vector database")
    parser.add_argument("--rag", action="store_true", help="Start interactive RAG session")
    parser.add_argument("--translate", action="store_true", help="Translate text")
    parser.add_argument("--summarize", action="store_true", help="Summarize text")
    parser.add_argument("--target-lang", default="en", choices=["en", "ar"], help="Target language for translation")
    parser.add_argument("--summary-strategy", default="abstractive", choices=["abstractive", "extractive"], help="Summarization strategy")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max characters for translation/summarization")
    args = parser.parse_args()
    main(args)