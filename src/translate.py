from langdetect import detect, LangDetectException
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm = OllamaLLM(model="llama3:8b", temperature=0.3)

def translate_text(text, target_lang="en", max_length=3500):
    try:
        source_lang = detect(text)
    except LangDetectException:
        return "Unable to detect source language."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    results = []

    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
        Translate and improve the fluency of the following text from {source_lang} to {target_lang}. 
        Maintain original meaning, structure, and tone. Return only the translated and refined version.
        strictly give the target text only.

        Text:
        \"\"\"{chunk}\"\"\"
        """
        try:
            translated = llm.invoke(prompt).strip()
        except Exception as e:
            translated = f"[Error in chunk {i}: {e}]"

        results.append(translated)

    return "\n\n".join(results)


if __name__ == "__main__":
    print("🌍 Language Translator with Fluency Enhancement (Ollama-powered)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text to translate: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break

        output = translate_text(user_input)
        print(f"\n✅ Translated & Refined:\n{output}\n")
