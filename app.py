import os
import streamlit as st
import PyPDF2
from transformers import pipeline
import requests
from io import BytesIO

# Ensure PyTorch is installed and loaded correctly
try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError:
    st.error("PyTorch is not installed. Please install it by running `pip install torch`.")
    raise

# Initialize a transformer-based pipeline for QA (RAG-style)
qa_model = pipeline("question-answering")

# Function to extract text from PDF (from URL)
def extract_text_from_pdf_from_url(pdf_url):
    """Extract text from a PDF file URL."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check if the request was successful
        file = BytesIO(response.content)
        
        # Read the PDF content using PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve the PDF: {e}")
        return ""
    except Exception as e:
        st.error(f"Failed to extract text from the PDF: {e}")
        return ""

# Function to get the PDF URL from GitHub repo
def get_pdf_url(doc_name):
    """Construct the URL for the PDF stored in the GitHub repository."""
    base_url = "https://raw.githubusercontent.com/Atif-Kazmi/companies-act-hackaton/blob/main/"
    return base_url + doc_name

# Streamlit UI for file selection and question input
st.title("Generative AI - Legal Query Assistant")

# Dropdown for document selection
documents = ["Companies Act 1984.pdf", "Income Tax Ordinance.pdf"]
selected_doc = st.selectbox("Select a document to query", documents)

# Construct the URL for the selected document
pdf_url = get_pdf_url(selected_doc)

# Extract text from the selected document via URL
document_text = extract_text_from_pdf_from_url(pdf_url)

# Text input for asking questions about the selected document
if document_text:
    user_question = st.text_input("Ask a question about the selected document:")
    if user_question:
        # Using the transformer model to answer questions
        try:
            answer = qa_model(question=user_question, context=document_text)
            st.write(f"Answer: {answer['answer']}")
        except Exception as e:
            st.error(f"Error processing your question: {e}")
else:
    st.warning("No document loaded. Please select a document to proceed.")

# Optional: Provide an upload option for users to add documents manually
uploaded_file = st.file_uploader("Or upload your own PDF", type="pdf")
if uploaded_file:
    # Save the uploaded file to the local environment (this will not save to GitHub)
    with open(f"uploaded_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"File uploaded: {uploaded_file.name}")
    
    # Extract text from the newly uploaded file
    uploaded_text = extract_text_from_pdf_from_url(f"uploaded_{uploaded_file.name}")
    
    if uploaded_text:
        user_question = st.text_input("Ask a question about the uploaded document:")
        if user_question:
            try:
                answer = qa_model(question=user_question, context=uploaded_text)
                st.write(f"Answer: {answer['answer']}")
            except Exception as e:
                st.error(f"Error processing your question: {e}")
else:
    st.info("You can upload your own PDF file if needed.")
