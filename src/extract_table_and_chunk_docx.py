import os
import json
from pathlib import Path
from docx import Document

def extract_and_chunk_tables_from_docx(file_path, rows_per_chunk=5, max_tokens=1000, overlap_tokens=100):
    """Extract and chunk tables from a .docx document, splitting by rows."""
    if not file_path.endswith('.docx'):
        return []

    try:
        doc = Document(file_path)
        chunks = []
        chunk_num = 1

        for table_idx, table in enumerate(doc.tables, 1):
            # Extract header
            header = [cell.text.strip().replace('\n', ' ') for cell in table.rows[0].cells]

            # Extract data rows
            data_rows = []
            for row in table.rows[1:]:
                seen = set()
                row_cells = []
                for cell in row.cells:
                    text = cell.text.strip().replace('\n', ' ')
                    if text not in seen:
                        seen.add(text)
                        row_cells.append(text)

                data_rows.append(row_cells)
                if any(cell.strip() for cell in row_cells):  # Only add non-empty rows
                    if row_cells not in data_rows:
                        data_rows.append(row_cells)

            # Create chunks with header and groups of rows
            for i in range(0, len(data_rows), rows_per_chunk):
                chunk_rows = data_rows[i:i + rows_per_chunk]
                chunks.append({
                    "file_name": os.path.basename(file_path),
                    "chunk_number": chunk_num,
                    "table_id": f"Table {table_idx}",
                    "header": header,
                    "rows": chunk_rows,
                    "section_type": "table"
                })
                chunk_num += 1

        return chunks

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_file(file_path, chunks_dir):
    """Process a file: extract tables if .docx, otherwise chunk text."""
    os.makedirs(chunks_dir, exist_ok=True)
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.docx':
        chunks = extract_and_chunk_tables_from_docx(file_path)

    if chunks:
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_tables" + ".json"
        output_path = os.path.join(chunks_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Chunked into {len(chunks)} pieces: {file_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    extracted_dir = base_dir / "outputs" / "extracted"
    chunks_dir = base_dir / "outputs" / "chunks"

    try:
        file_path = "D:\projects\dr.x-assignment\data\M.Sc. Applied Psychology.docx"
        print(f"Processing: {file_path}")
        process_file(file_path, str(chunks_dir))
    except Exception as e:
        print(f"Error accessing directory {extracted_dir}: {e}")