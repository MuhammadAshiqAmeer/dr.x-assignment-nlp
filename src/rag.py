import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable

class RAGSystem:
    def __init__(self, vector_db_path="../outputs/vector_db"):
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Load FAISS vector store
        self.vector_store = FAISS.load_local(
            vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Initialize LLM
        self.llm = OllamaLLM(model="llama3:8b")

        # Define prompt template
        self.prompt = PromptTemplate.from_template(
            """You are an intelligent assistant designed to answer questions **based on the provided context only**.

            Context:
            {context}

            Instructions:
            - Do **not** assume the identity of the person in the context (e.g., resume).
            - If the user greets (e.g., "hello", "hi"), respond politely (maximum 10 words).

            User: {question}
            Assistant:"""
        )

        # Create chain
        self.chain: Runnable = self.prompt | self.llm
        self.history = []

    def query(self, question: str) -> str:
        # Perform similarity search
        docs = self.vector_store.similarity_search(question, k=5)
        
        # Extract contexts and metadata for debugging
        contexts = []
        for i, doc in enumerate(docs):
            context = doc.page_content
            contexts.append(context)
            # Debug retrieved contexts (uncomment to enable)
            # print(f"Context {i+1}: {context[:100]}... (metadata: {doc.metadata})")

        # Combine contexts
        context_str = "\n".join(contexts)

        # Invoke LLM
        result = self.chain.invoke({
            "context": context_str,
            "question": question
        })
        answer = result

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

if __name__ == "__main__":
    # Initialize RAG system with path to vector store
    rag = RAGSystem(vector_db_path=Path(__file__).parent.parent / "outputs" / "vector_db")
    print("🔍 RAG QA System Ready. Type 'exit' to quit.\n")

    while True:
        question = input("🧠 You: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        answer = rag.query(question)
        print(f"\n🤖 Assistant: {answer}\n")