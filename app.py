import streamlit as st
import logging
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import PyPDF2
from io import BytesIO

# Suppress transformer-related warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load the BERT model and tokenizer for question answering
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Create a pipeline for QA
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Streamlit UI
st.title('Legal Document Question Answering')

st.markdown("""
This app allows you to upload legal documents (e.g., Companies Act 1984) in PDF format
and ask questions related to the document. The model used is BERT-based, fine-tuned on 
the SQuAD dataset, which allows it to answer questions based on the uploaded document.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("Document successfully uploaded and text extracted.")

    # Allow the user to ask questions
    question = st.text_input("Ask a question based on the document:")

    if question:
        # Get the answer from the model
        result = qa_pipeline({
            'context': document_text,
            'question': question
        })

        # Display the answer
        st.write(f"Answer: {result['answer']}")

# Example instructions
st.markdown("""
### Example Questions:
- "What are the key responsibilities of directors?"
- "What is the procedure for company registration?"
- "Explain the rules for dissolution of a company."
""")
