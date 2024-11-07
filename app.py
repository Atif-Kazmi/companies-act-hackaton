import os
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile

# Function to load and read PDF from URL
def load_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return None

# Function to handle file uploads
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load your specific model for question-answering
model_name = "bert-base-uncased"  # Replace with your desired model
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Streamlit UI setup
st.title("Legal Document Question Answering")

# Upload section for PDF file
uploaded_file = st.file_uploader("Upload Companies Act 1984 or Income Tax Ordinance PDF", type=["pdf"])

if uploaded_file is not None:
    # If file uploaded, read the contents
    document_text = read_pdf(uploaded_file)
    st.success("PDF file loaded successfully!")
else:
    st.warning("Please upload a PDF file to proceed.")

# Option to input question
question = st.text_input("Ask a question based on the uploaded document:")

# Handle question and answer pipeline
if question and document_text:
    try:
        # Use the QA pipeline to answer the question based on the document
        answer = qa_pipeline(question=question, context=document_text)
        st.write(f"Answer: {answer['answer']}")
    except Exception as e:
        st.error(f"Error in processing: {e}")

# Option to load a default PDF from GitHub if no file uploaded
if uploaded_file is None:
    st.info("You can also try with the default Companies Act 1984 PDF from GitHub")
    if st.button("Load Default Companies Act PDF"):
        pdf_url = "https://raw.githubusercontent.com/Atif-Kazmi/companies-act-hackaton/main/Companies%20Act%201984.pdf"
        document_text = load_pdf_from_url(pdf_url)
        if document_text:
            st.success("Default Companies Act PDF loaded successfully!")
        else:
            st.error("Failed to retrieve PDF from GitHub.")

# Instructions for the user
st.sidebar.markdown("""
    ## How to Use:
    1. Upload a PDF file (Companies Act 1984 or Income Tax Ordinance).
    2. Ask a specific question related to the document.
    3. The AI model will try to find the answer from the document and display it.
""")

