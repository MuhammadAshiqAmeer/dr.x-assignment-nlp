import faiss
import json
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable
from ollama import Client

class RAGSystem:
    def __init__(self, index_path="../outputs/vector_db.faiss", metadata_path="../outputs/metadata.json"):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.llm = OllamaLLM(model="llama3:8b")
        self.embedder = Client()

        self.prompt = PromptTemplate.from_template(
        """You are an intelligent assistant designed to answer questions **based on the provided context only**.

        Context:
        {context}

        Instructions:
        - Do **not** assume the identity of the person in the context (e.g., resume).
        - If the user greets (e.g., "hello", "hi"), respond politely(maximum 15 words).").

        User: {question}
        Assistant:"""
        )

        self.chain: Runnable = self.prompt | self.llm
        self.history = []

    def query(self, question: str) -> str:
        # Embed the question
        embedding_response = self.embedder.embeddings(model="nomic-embed-text", prompt=question)
        query_embedding = np.array([embedding_response["embedding"]], dtype=np.float32)

        # Search FAISS
        distances, indices = self.index.search(query_embedding, k=5)
        
        contexts = [self.metadata[i]["text"] for i in indices[0]]

        # Debug retrieved contexts
        # print(f"\nRetrieved contexts for '{question}':")
        # for i, ctx in enumerate(contexts):
        #     print(f"Context {i+1}: {ctx[:100]}... (distance: {distances[0][i]})")
        # print('\n')

        
        # Invoke LLM
        result = self.chain.invoke({
            "context": "\n".join(contexts),
            "question": question
        })
        answer = result

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer


if __name__ == "__main__":
    rag = RAGSystem()
    print("🔍 RAG QA System Ready. Type 'exit' to quit.\n")

    while True:
        question = input("🧠 You: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        answer = rag.query(question)
        print(f"\n🤖 Assistant: {answer}\n")