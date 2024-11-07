import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import PyPDF2
import streamlit as st

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Set up the model and tokenizer for DistilBERT (fine-tuned for QA)
model_name = "distilbert-base-cased-distilled-squad"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Function to answer questions using the QA model
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app interface
def main():
    st.title("Legal Document QA Assistant")

    # Upload PDF
    uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")

    if uploaded_file is not None:
        # Save the file temporarily
        pdf_path = "/tmp/uploaded_document.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract text from the uploaded PDF
        context = extract_pdf_text(pdf_path)

        # Display a message after extraction
        st.write("Text extraction complete. You can now ask questions based on the document.")

        # Get user input for a question
        question = st.text_input("Ask a question about the document:")

        if question:
            # Get the answer from the model
            answer = answer_question(question, context)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please ask a question.")

if __name__ == "__main__":
    main()
