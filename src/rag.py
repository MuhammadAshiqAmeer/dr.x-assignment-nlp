import faiss
import numpy as np
from ollama import Client
import json

class RAGSystem:
    def __init__(self, index_path="../outputs/vector_db.faiss", metadata_path="../outputs/metadata.json"):
        self.ollama = Client()
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.history = []

    def query(self, question):
        # Embed the user's question
        response = self.ollama.embeddings(model="nomic-embed-text", prompt=question)
        query_embedding = np.array([response["embedding"]], dtype=np.float32)

        # Search the FAISS vector DB
        distances, indices = self.index.search(query_embedding, k=5)
        contexts = [self.metadata[i]["text"] for i in indices[0]]

        # Build a cleaner, chat-style prompt
        prompt = f"""You are a helpful assistant. Use the provided context to answer the user's question. Donot hallucinate.

                    Context:
                    {chr(10).join(contexts)}

                    <|user|>
                    {question}

                    <|assistant|>
        """

        # Generate response using a small LLM (you can change model name here)
        response = self.ollama.generate(
            model="gemma3:1b",  # or "gemma3:12b" or bigger models based on your system performance and storage
            prompt=prompt,
        )
        answer = response["response"]

        # Track conversation history (optional for future extensions)
        self.history.append({
            "role": "user",
            "content": question
        })
        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer

if __name__ == "__main__":
    rag = RAGSystem()
    print("🔍 RAG QA System Ready. Type 'exit' to quit.\n")
    
    while True:
        question = input("🧠 Ask a question: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        answer = rag.query(question)
        print(f"\n📘 Answer: {answer}\n")
