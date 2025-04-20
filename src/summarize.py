from langchain_ollama import OllamaLLM
from rouge_score import rouge_scorer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm


llm = OllamaLLM(model="llama3:8b", temperature=0.3)

def split_text(text, max_length=8000):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def summarize_chunk(chunk, strategy="abstractive"):
    chunk = chunk.strip()
    if not chunk:
        return ""

    if strategy == "abstractive":
        prompt = f"""
        You are a highly intelligent summarization system. strictly give the target text only.

        Summarize the following passage into a clear, structured summary, identifying and including:
        
        - Key points, findings, or events
        - Main subjects (e.g. people, concepts, or entities)
        - Purpose or objective of the text (if relevant)
        - Important conclusions, outcomes, or messages

        Adapt  and add your summary based on the content type (e.g., story, research, report, article) — be concise, accurate, and faithful to the source.

        Passage:
        \"\"\"{chunk}\"\"\"

        Structured Summary:
        """
    else:
        prompt = f"Extract the most important sentences from this passage:\n\n{chunk}\n\nExtracted Summary:"

    return llm.invoke(prompt).strip()


def recursive_summarize(texts, strategy="abstractive", max_length=8000, depth=0, max_depth=10):
    print(f"\n📚 Summarizing {len(texts)} chunks at depth {depth}...")
    summaries = [summarize_chunk(text, strategy) for text in tqdm(texts) if text.strip()]
    combined = "\n\n".join(summaries)

    if len(combined) > max_length and depth < max_depth:
        chunks = split_text(combined, max_length)
        return recursive_summarize(chunks, strategy, max_length, depth + 1, max_depth)
    else:
        return summarize_chunk(combined, strategy)

def summarize_text(text, strategy="abstractive", max_length=8000):
    chunks = split_text(text, max_length)
    return recursive_summarize(chunks, strategy, max_length)

def evaluate_summary(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)

if __name__ == "__main__":
    from extract_text import extract_text_from_file

    print("📖 Loading file")
    raw_text = extract_text_from_file("../data/The-Alchemist.pdf")

    print("🧹 Cleaning and preparing text...")
    summary = summarize_text(raw_text, strategy="extractive")

    print(f"\n📝 Final Summary:\n{summary}\n")
