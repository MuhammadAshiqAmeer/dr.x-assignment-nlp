from langdetect import detect, LangDetectException
from langchain_ollama import OllamaLLM

def translate_text(text, target_lang="en"):
    """Translate text to target language using Ollama and improve fluency."""
    try:
        source_lang = detect(text)
    except LangDetectException:
        return "Unable to detect source language."

    llm = OllamaLLM(model="llama3:8b")

    # Step 1: Translate
    translate_prompt = f"Translate the following text from {source_lang} to {target_lang}, maintaining structure and fluency:\n\n{text}"
    try:
        translated_text = llm.invoke(translate_prompt).strip()
    except Exception as e:
        return f"Translation failed: {e}"

    # Step 2: Improve fluency
    fluency_prompt = f"Improve the grammar and fluency of this text:\n\n{translated_text}"
    try:
        improved_text = llm.invoke(fluency_prompt).strip()
    except Exception as e:
        return f" Fluency enhancement failed: {e}"

    return improved_text


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
