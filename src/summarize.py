from langchain_ollama import OllamaLLM
from rouge_score import rouge_scorer
import tiktoken

def recursive_summarize(texts, strategy="abstractive", max_tokens=3500):
    """Recursively summarize a list of texts until they fit into a final summary."""
    llm = OllamaLLM(model="llama3:8b")
    encoding = tiktoken.get_encoding("cl100k_base")

    def summarize_chunk(chunk):
        if strategy == "abstractive":
            prompt = f"Provide a concise summary of the following text, capturing the key ideas. Strictly provide the abstrated text only:\n\n{chunk}"
        else:
            prompt = f"Extract the most important sentences from the following text. Strictly provide the extracted text only:\n\n{chunk}"
        return llm.invoke(prompt).strip()

    # Step 1: Summarize each chunk
    summaries = []
    for text in texts:
        summaries.append(summarize_chunk(text))

    # Step 2: If the combined summary is too long, repeat recursively
    combined = "\n\n".join(summaries)
    if len(encoding.encode(combined)) > max_tokens:
        # Chunk again and recursively summarize
        return recursive_summarize(split_text(combined, max_tokens), strategy, max_tokens)
    else:
        # Final summary
        return summarize_chunk(combined)

def split_text(text, max_tokens=3500):
    """Split a long text into token-based chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
    return chunks

def summarize_text(text, strategy="abstractive", max_tokens=3500):
    """Entry point for summarizing long text."""
    chunks = split_text(text, max_tokens=max_tokens)
    return recursive_summarize(chunks, strategy, max_tokens)

def evaluate_summary(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)



if __name__ == "__main__":
    print("Summarization and Evaluation System Ready.")
    
    # Example text (replace with your own text to test)
    text_to_summarize = """
    Stable isotope analysis (SIA) is used extensively in marine and ecological studies.
    The method uses isotopic compositions of elements such as carbon (δ13C), nitrogen (δ15N), and oxygen (δ18O) to study various aspects like animal movement, trophic dynamics, and ecosystem function.
    The use of stable isotopes in oceanography helps identify the sources of organic carbon in marine ecosystems and track nutrient pathways.
    """

    # Summarize the text
    summary = summarize_text(text_to_summarize)
    print(f"Summary: {summary}\n")
    
    # Evaluate summary (example reference)
    reference_text = "This study uses stable isotope analysis to track the movement and trophic interactions of marine organisms, focusing on isotopes like δ13C and δ15N."
    evaluation_scores = evaluate_summary(reference_text, summary)
    print(f"ROUGE Scores: {evaluation_scores}")