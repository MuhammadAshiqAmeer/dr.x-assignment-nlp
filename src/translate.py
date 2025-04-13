from langdetect import detect, LangDetectException
from langchain_ollama import OllamaLLM
import tiktoken

def translate_text(text, target_lang="en", max_tokens=3500):
    """Translate long text by chunking and improving fluency per chunk."""
    try:
        source_lang = detect(text)
    except LangDetectException:
        return "Unable to detect source language."

    llm = OllamaLLM(model="llama3:8b")
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    translated_chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)

        # Step 1: Translation
        translate_prompt = (
            f"Translate the following text from {source_lang} to {target_lang}, maintaining structure and fluency. "
            f"Strictly return only the translated text.\n\n{chunk_text}"
        )
        try:
            translated = llm.invoke(translate_prompt).strip()
        except Exception as e:
            translated = f"[Translation failed: {e}]"

        # Step 2: Fluency improvement
        fluency_prompt = (
            f"Improve the grammar and fluency of this translated text from {source_lang} to {target_lang}. "
            f"Strictly return only the improved text.\n\n{translated}"
        )
        try:
            improved = llm.invoke(fluency_prompt).strip()
        except Exception as e:
            improved = f"[Fluency improvement failed: {e}]"

        translated_chunks.append(improved)

    return "\n\n".join(translated_chunks)



if __name__ == "__main__":
    print("Language Translator with Fluency Enhancement (Ollama-powered)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text to translate: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        result = translate_text(user_input)
        print(f"\nTranslated & Improved:\n{result}\n")
