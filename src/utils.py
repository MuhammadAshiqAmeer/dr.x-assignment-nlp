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
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(text))

    start_time = time.time()
    result = task_func(text)
    elapsed_time = time.time() - start_time
    tokens_per_second = tokens / elapsed_time if elapsed_time > 0 else 0

    performance_file = "outputs/performance.json"
    try:
        if os.path.exists(performance_file):
            with open(performance_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except json.JSONDecodeError:
        data = []

    data.append({task_name: {"tokens_per_second": tokens_per_second}})

    with open(performance_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return result

