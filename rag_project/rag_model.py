from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar el modelo LLM de Ollama con Gemma:2b
llm = Ollama(model="gemma:2b")

def generate_embeddings(text_chunks):
    """Genera embeddings para los fragmentos de texto usando OllamaEmbeddings con Gemma."""
    embeddings = OllamaEmbeddings(model="gemma:2b")
    return FAISS.from_texts(text_chunks, embeddings)

def create_vectorstore(text):
    """Divide el texto en fragmentos y lo almacena en un vectorstore"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_text(text)
    return generate_embeddings(text_chunks)

def answer_question(vectorstore, question):
    """Busca en la base de conocimiento y responde preguntas usando RAG"""
    docs = vectorstore.similarity_search(question, k=3)  # Recuperar los 3 documentos más relevantes
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Contexto: {context}
    Pregunta: {question}
    Responde de manera precisa basándote en el contexto.
    """
    
    return llm.invoke(prompt)
