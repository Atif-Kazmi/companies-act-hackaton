import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
from io import BytesIO

# Step 1: Extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 2: Convert text into embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Setup the generative model (GPT-2 or GPT-Neo)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Function to perform the similarity search
def search(query, sections, index):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=3)  # Top 3 relevant sections
    return [sections[idx] for idx in indices[0]]

# Function to generate an answer using the retrieved context
def generate_answer(query, relevant_sections):
    context = " ".join(relevant_sections)  # Concatenate the retrieved sections
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    answer = generator(input_text, max_length=150, num_return_sequences=1)
    return answer[0]['generated_text']

# Streamlit UI setup
st.title("Companies Act 1984 Query Assistant")
st.write("Upload the Companies Act 1984 PDF to ask any question related to the document.")

# Upload file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    sections = pdf_text.split("\n")  # Split the text into sections or paragraphs
    embeddings = model.encode(sections)

    # Create a FAISS index for similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(np.array(embeddings, dtype=np.float32))

    # User input for query
    query = st.text_input("Enter your query:")

    if query:
        relevant_sections = search(query, sections, index)  # Search for relevant sections
        answer = generate_answer(query, relevant_sections)  # Generate an answer
        st.write("**Answer:**")
        st.write(answer)  # Display the generated answer
