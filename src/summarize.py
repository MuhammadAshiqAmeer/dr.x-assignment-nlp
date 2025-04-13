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
