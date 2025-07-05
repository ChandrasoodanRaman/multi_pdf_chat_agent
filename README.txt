📚 Multi-PDFs ChatApp AI Agent 🤖
Meet the MultiPDF Chat AI App! 🚀 Chat seamlessly with multiple PDFs using LangChain, Google Gemini Pro, and FAISS — deployed via Streamlit. Ask questions and get accurate, contextual answers from your documents instantly. 🔥

📝 Description
This is a Streamlit-based application that allows users to upload multiple PDFs, extract and embed their content, and ask natural language questions about them using Google Gemini. It uses chunked embeddings and semantic search for retrieval-augmented generation (RAG).

🎯 Key Features
📄 Multi-PDF Upload & Processing

🧠 Google Gemini Pro LLM Integration

🧩 Text Chunking with RecursiveCharacterTextSplitter

🔎 Vector Search using FAISS

💬 Conversational QA Chain with Context

🧾 Supports both PDF & TXT formats

🧠 Model-agnostic architecture (Gemini, GPT, Claude, LLaMA)

⚙️ How It Works
PDF Loading: Reads uploaded PDFs and extracts raw text.

Text Chunking: Breaks long text into manageable chunks.

Embeddings: Converts chunks into dense vector embeddings.

Similarity Search: Finds most relevant chunks per query.

Response Generation: Gemini Pro uses retrieved content to answer user queries.