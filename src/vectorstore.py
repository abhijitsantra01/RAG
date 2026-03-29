import os
import faiss
import numpy as np
import pickle
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # 👇 Add the type hint here
        self.index: Optional[faiss.Index] = None 
        
        self.metadata = []
    

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ========================
    # Build Index
    # ========================
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # 🔥 Better metadata (VERY IMPORTANT)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "unknown")
            }
            for chunk in chunks
        ]

        embeddings = np.array(embeddings).astype("float32")

        # ✅ Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.add_embeddings(embeddings, metadatas)
        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    # ========================
    # Add embeddings
    # ========================
    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Any]] = None):
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # 🔥 cosine similarity

        self.index.add(embeddings)  #type:ignore

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors.")

    # ========================
    # Save / Load
    # ========================
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved index to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError("❌ FAISS index not found")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded index from {self.persist_dir}")

    # ========================
    # Search
    # ========================
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("❌ Index not loaded. Call load() first.")

        # Normalize query too
        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(query_embedding, top_k)  # type: ignore

        results = []
        for idx, score in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None

            results.append({
                "index": idx,
                "score": float(score),  # similarity score
                "metadata": meta
            })

        return results

    # ========================
    # Query
    # ========================
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Query: {query_text}")

        query_emb = self.model.encode([query_text]).astype("float32")

        return self.search(query_emb, top_k=top_k)


# ========================
# Example
# ========================
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()

    results = store.query("What is attention mechanism?", top_k=3)

    for r in results:
        print("\n---")
        print("Score:", r["score"])
        print("Source:", r["metadata"]["source"])
        print("Page:", r["metadata"]["page"])
        print("Text:", r["metadata"]["text"][:200])