import streamlit as st
from pdf_processor import extract_text_from_pdf
from rag_model import create_vectorstore, answer_question

st.title("📚 Sistema de Preguntas y Respuestas con RAG 🤖")

# Subida de archivos PDF
uploaded_file = st.file_uploader("📂 Sube un archivo PDF", type="pdf")

if uploaded_file is not None:
    with open(f"docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extraer texto del PDF
    text = extract_text_from_pdf(f"docs/{uploaded_file.name}")
    
    # Crear el vectorstore con embeddings
    vectorstore = create_vectorstore(text)
    
    st.success("📄 Documento procesado exitosamente. ¡Ahora puedes hacer preguntas!")

    # Cuadro de entrada para preguntas del usuario
    question = st.text_input("❓ Escribe tu pregunta:")

    if question:
        response = answer_question(vectorstore, question)
        st.write("🤖 **Respuesta:**", response)
