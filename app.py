import os
import streamlit as st
import PyPDF2
from groq import Groq

# Set up Groq API client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Streamlit UI setup
st.title('Legal Document Question Answering with Groq API')

st.markdown("""
This app allows you to upload legal documents (e.g., Companies Act 1984) in PDF format
and ask questions related to the document using Groq's language model.
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
        # Use Groq API to get the answer based on the uploaded document text
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Answer the following question based on the document: {question}"},
                {"role": "system", "content": f"The context is: {document_text}"},
            ],
            model="llama3-8b-8192",
        )

        # Display the answer from the Groq model
        st.write(f"Answer: {chat_completion.choices[0].message.content}")

# Example instructions
st.markdown("""
### Example Questions:
- "What are the key responsibilities of directors?"
- "What is the procedure for company registration?"
- "Explain the rules for dissolution of a company."
""")
