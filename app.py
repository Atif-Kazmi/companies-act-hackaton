import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import os

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
generator = pipeline('text-generation', model='gpt2')  # Use GPT-2 for faster response

# Function to perform the similarity search and limit the number of sections
def search(query, sections, index):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=5)  # Limit to top 5 sections
    return [sections[idx] for idx in indices[0]]

# Function to generate an answer using the retrieved context
def generate_answer(query, relevant_sections):
    context = " ".join(relevant_sections[:5])  # Limit to top 5 sections for faster processing
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    
    # Generate answer with truncation and optimized max_length
    answer = generator(input_text, max_length=100, num_return_sequences=1, truncation=True)
    return answer[0]['generated_text']

# Function to load and process a PDF from the repo
def load_document(doc_name):
    file_path = os.path.join("documents", doc_name)  # Assuming PDFs are in a 'documents' folder
    if os.path.exists(file_path):
        return extract_text_from_pdf(file_path)
    else:
        return None

# Streamlit UI setup
st.title("Document Query Assistant")
st.write("Select a document and ask a question.")

# Dropdown to select document
doc_choice = st.selectbox("Choose a document:", ["Companies Act 1984.pdf", "Income Tax Ordinance.pdf"])

# Load the selected document
pdf_text = load_document(doc_choice)
if pdf_text:
    sections = pdf_text.split("\n")  # Split the text into sections or paragraphs
    embeddings = model.encode(sections)

    # Create a FAISS index for similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(np.array(embeddings, dtype=np.float32))

    # User input for query
    query = st.text_input("Enter your query:")

    if query:
        # Retrieve relevant sections from the document
        relevant_sections = search(query, sections, index)
        # Generate an answer using the relevant sections
        answer = generate_answer(query, relevant_sections)
        st.write("**Answer:**")
        st.write(answer)  # Display the generated answer
else:
    st.write(f"Sorry, the document '{doc_choice}' could not be found.")
