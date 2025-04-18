import os
from pathlib import Path
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
)
from langchain_community.document_loaders.pdf import PyMuPDFLoader

if os.path.basename(os.getcwd()) == "src":
    from utils import save_text
    output_dir = Path("../outputs/extracted")
else:
    from src.utils import save_text
    output_dir = Path("outputs/extracted")

def extract_text_from_file(file_path):
        """Extract text from various file formats using LangChain document loaders."""
    # try:
        extension = os.path.splitext(file_path)[1].lower()
        text_output = []

        if extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            text_output.extend([doc.page_content for doc in docs])

        elif extension == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            text_output.extend([doc.page_content for doc in docs])

        elif extension == '.pdf':
            # Standard text extraction
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                page_num = doc.metadata.get('page', 1) + 1
                text_output.append(f"Page {page_num}: {doc.page_content}")

            # Figure caption extraction using PyMuPDF
            caption_loader = PyMuPDFLoader(file_path)
            docs = caption_loader.load()
            for doc in docs:
                page_num = doc.metadata.get('page', 1) + 1
                text = doc.page_content.strip()
                if text.lower().startswith("figure"):
                    caption = f"[Figure Caption - Page {page_num}]: {text}"
                    text_output.append(caption)

        elif extension == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
            docs = loader.load()
            text_output.extend([doc.page_content for doc in docs])

        elif extension in ['.xlsx', '.xls', '.xlsm']:
            # Explicitly use UnstructuredExcelLoader for Excel files
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            docs = loader.load()
            text_output.extend([doc.page_content for doc in docs])

        else:
            raise ValueError(f"Unsupported file format: {extension}")

        extracted_text = '\n'.join([t for t in text_output if t.strip()])
        relative_name = Path(file_path).stem + ".txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / relative_name

        save_text(extracted_text, str(save_path))
        return extracted_text

    # except Exception as e:
    #     print(f"Error processing {file_path}: {e}")
    #     return None

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    if not data_dir.exists():
        print(f"Directory '{data_dir}' does not exist.")
    else:
        for file_path in data_dir.rglob("*.*"):
            if file_path.is_file():
                print(f"Processing: {file_path}")
                extracted = extract_text_from_file(file_path)
                if extracted:
                    print(f"Extraction successful: {file_path.name}")
                else:
                    print(f"Failed to extract text from: {file_path.name}")