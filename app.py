import os
import streamlit as st
from groq import Groq

# Access API key securely using Streamlit's Secrets Manager
api_key = st.secrets["groq"]["GROQ_API_KEY"]

# Initialize the Groq client with the API key
client = Groq(api_key=api_key)

# Function to get Groq API response
def get_groq_response(question):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": question,
        }],
        model="llama3-8b-8192",  # Change model to your preferred choice
    )
    return chat_completion.choices[0].message.content

# Streamlit app layout
st.title("Groq API Chatbot")
st.write("Ask me anything related to the Companies Act or Income Tax Ordinance.")

# Get user input
question = st.text_input("Ask a Question:")

if question:
    # Get the response from Groq API
    response = get_groq_response(question)
    st.write("Answer: ", response)
else:
    st.write("Please enter a question to get an answer.")
