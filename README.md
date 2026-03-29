# 📚 RAG-based PDF Question Answering System

A **Retrieval-Augmented Generation (RAG)** system that allows you to ask questions over your PDF documents and get accurate, context-aware answers using **FAISS + Sentence Transformers + LLM (Gemini)**.

---

## 🚀 Features

- 📄 Load and process multiple PDF documents
- ✂️ Intelligent text chunking with overlap
- 🧠 Semantic embeddings using `sentence-transformers`
- 🔎 Fast similarity search using `FAISS`
- 🤖 LLM-powered answers (Gemini)
- 💾 Persistent vector database (save & reload)
- 📑 Source-aware responses (file + page support)
- ⚡ Modular and scalable architecture

---

## 🧠 Architecture
PDFs → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer


---

## 📁 Project Structure


RAG/
│
├── data/
│ └── pdf_files/ # Input PDFs
│
├── vector_store/ # Saved FAISS index
│
├── src/
│ ├── data_loader.py # Load PDFs
│ ├── embedding.py # Chunking + embeddings
│ ├── vectorstore.py # FAISS logic
│ ├── rag.py # RAG pipeline
│ └── __init__.py
│
├── main.py # Entry point
├── requirements.txt
├── README.md


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa
2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3. Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory:

For OpenAI:
OPENAI_API_KEY=your_api_key_here
▶️ Usage
1. Add your PDFs

Place all your PDF files inside:

data/pdf_files/
2. Run the application
python main.py
3. Ask questions
>> What is attention mechanism?

🧠 Answer:
The attention mechanism allows models to focus on relevant parts...
🧠 Supported Models
Embedding Models
all-MiniLM-L6-v2 (default, fast)
all-mpnet-base-v2 (higher accuracy)
LLM Options
OpenAI (e.g., gpt-4o-mini)
Local models like Phi-3 Mini
⚡ How It Works
📄 PDFs are loaded and split into chunks
🧠 Each chunk is converted into embeddings
🔎 FAISS stores vectors for fast retrieval
❓ User query is embedded
🎯 Top-K relevant chunks are retrieved
🤖 LLM generates answer using context
📊 Example Output
Answer:
The attention mechanism is a technique that allows models to focus on important parts of input data.

Sources:
- attention.pdf (Page 3)
- transformers.pdf (Page 7)
🛠️ Tech Stack
Python
LangChain
Sentence Transformers
FAISS
OpenAI API / Phi-3 Mini
NumPy
🔥 Future Improvements
✅ Streamlit / Web UI
✅ Hybrid search (BM25 + FAISS)
✅ Reranking models
✅ Chat history memory
✅ Streaming responses
✅ Docker deployment
🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a PR.

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgements
HuggingFace
FAISS (Facebook AI)
LangChain
OpenAI
💡 Author
Abhijit Santra
GitHub:https://github.com/abhijitsantra01
