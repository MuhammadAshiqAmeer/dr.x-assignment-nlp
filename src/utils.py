import os
import time
import tiktoken
import json

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


def measure_performance(text, task_func, task_name):
    """Measure tokens per second for a task."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(text))
    start_time = time.time()
    result = task_func(text)
    elapsed_time = time.time() - start_time
    tokens_per_second = tokens / elapsed_time if elapsed_time > 0 else 0
    
    with open("outputs/performance.json", "a") as f:
        json.dump({task_name: {"tokens_per_second": tokens_per_second}}, f)
    return result