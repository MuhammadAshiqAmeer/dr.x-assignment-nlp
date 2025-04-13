import os

def save_text(text, output_path):
    """Save text to a file, creating directories if needed."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save text with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Error saving text to {output_path}: {e}")

