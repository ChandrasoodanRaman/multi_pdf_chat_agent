import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API Key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_MODEL = "models/embedding-001"      # Embedding model for vector store
CHAT_MODEL = "models/gemini-2.5-pro"          # Chat model for Q&A

# 1. Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# 2. Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

# 3. Convert text chunks into embeddings and store in FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# 4. Build the Q&A chain with Gemini
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, respond with: "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 5. Handle user query

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists("faiss_index/index.faiss") or not os.path.exists("faiss_index/index.pkl"):
        st.warning("⚠️ Please upload and process PDFs first to build the knowledge base.")
        return
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("📣 **Reply:**", response["output_text"])


    

# 6. Streamlit UI
def main():
    st.set_page_config("Multi PDF Chatbot", page_icon="📚")
    st.header("📚 Multi-PDF Chat Agent 🤖")

    user_question = st.text_input("💬 Ask a question about your uploaded PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("📤 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
