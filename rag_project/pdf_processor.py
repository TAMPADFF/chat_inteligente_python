import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un archivo PDF usando pdfplumber"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text
