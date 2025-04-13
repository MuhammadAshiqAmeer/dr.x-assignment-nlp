from langchain_ollama import OllamaLLM
from rouge_score import rouge_scorer

def summarize_text(text, strategy="abstractive"):
    """Summarize text using Ollama LLM."""
    llm = OllamaLLM(model="llama3:8b")
    
    if strategy == "abstractive":
        prompt = f"Provide a concise summary of the following text, capturing the key ideas:\n\n{text}"
    else:
        prompt = f"Extract the most important sentences from the following text:\n\n{text}"
    
    try:
        summary = llm.invoke(prompt).strip()
    except Exception as e:
        return f" Summarization failed: {e}"
    
    return summary

def evaluate_summary(reference, summary):
    """Evaluate the summary using ROUGE metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores


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