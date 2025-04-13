import os
from docx import Document
import PyPDF2
import pandas as pd
from utils import save_text

def extract_text_from_file(file_path):
    """Extract text from various file formats."""
    try:
        extension = os.path.splitext(file_path)[1].lower()
        text_output = []
        
        if extension == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text_output.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                    text_output.append(row_text)
        
        elif extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_output.append(f"Page {page_num + 1}: {page.extract_text()}")
        
        elif extension in ['.csv', '.xlsx', '.xls', '.xlsm']:
            if extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            text_output.append(df.to_string(index=False))
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        extracted_text = '\n'.join([t for t in text_output if t.strip()])
        save_text(extracted_text, file_path.replace('data', 'outputs/extracted'))
        return extracted_text
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    

if __name__ == "__main__":
    data_dir = os.path.join("..",'data')
    
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' does not exist.")
    else:
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                extracted = extract_text_from_file(file_path)
                if extracted:
                    print(f"Extraction successful: {file}")
                else:
                    print(f"Failed to extract text from: {file}")
