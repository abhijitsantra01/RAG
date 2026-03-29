import os
import requests
from dotenv import load_dotenv

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "phi3*mini"
    ):
        print("[INFO] Initializing RAG system...")

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # ========================
        # Load or Build Vector DB
        # ========================
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[INFO] No existing vector store found. Building...")
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            print("[INFO] Loading existing vector store...")
            self.vectorstore.load()

        self.llm_model = llm_model
        print(f"[INFO] Using local LLM via Ollama: {llm_model}")

    # ========================
    # Call Phi-3 via Ollama
    # ========================
    def _call_llm(self, prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            return response.json()["response"]

        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

    # ========================
    # RAG Query
    # ========================
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        print(f"[INFO] Searching for: {query}")

        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context.strip():
            return "No relevant documents found."

        # 🔥 Better prompt
        prompt = f"""
You are an AI assistant. Answer ONLY from the context below.

If the answer is not present, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        return self._call_llm(prompt)


# ========================
# Run Script
# ========================
if __name__ == "__main__":
    rag_search = RAGSearch()

    print("\n✅ RAG system ready! Type 'exit' to quit.\n")

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        answer = rag_search.search_and_summarize(query, top_k=3)

        print("\n🧠 Answer:\n", answer, "\n")