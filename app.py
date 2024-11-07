import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st

# Step 1: Extract text from the Companies Act PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Convert text into embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Load the PDF and create embeddings
pdf_path = "Companies_Act_1984.pdf"  # Path to your Companies Act PDF
pdf_text = extract_text_from_pdf(pdf_path)
sections = pdf_text.split("\n")  # Split the text into sections or paragraphs
embeddings = model.encode(sections)

# Step 4: Create a FAISS index for similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
index.add(np.array(embeddings, dtype=np.float32))

# Step 5: Setup the generative model (GPT-2 or GPT-Neo)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Function to perform the similarity search
def search(query):
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
st.write("Ask any question related to the Companies Act 1984, and I will provide an answer from the document.")

# User input
query = st.text_input("Enter your query:")

if query:
    relevant_sections = search(query)  # Search for relevant sections
    answer = generate_answer(query, relevant_sections)  # Generate an answer
    st.write("**Answer:**")
    st.write(answer)  # Display the generated answer
