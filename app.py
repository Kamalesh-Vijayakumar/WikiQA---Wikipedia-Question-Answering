from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from typing import List
import numpy as np
import os
import google.generativeai as genai

app = FastAPI()



# Define the request models
class LoadDataRequest(BaseModel):
    url: str

class QueryDataRequest(BaseModel):
    query: str

# Initialize MongoDB Connection
client = MongoClient('localhost', 27017)
db = client['vector_db']
collection = db['text_vectors']

# Initialize Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Gemini API
os.environ["GEMINI_API_KEY"] = "enter your api key"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    As a prompt engineer, the goal is to improve and expand the given text to make it more detailed and informative. You are provided with a summary or a brief response that needs enhancement. Your task is to rewrite the text to include more context, background information, and any other relevant details that will help the user better understand the content.

    ### Input Example:
    "An LLM is a language model, which is not an agent as it has no goal, but it can be used as a component of an intelligent agent."

    ### Output Example:
    "An LLM, or Large Language Model, is a sophisticated type of machine learning model designed to understand and generate human-like text. Although an LLM itself is not an autonomous agent and lacks goals or motivations, it plays a crucial role in natural language processing tasks. For instance, when integrated into intelligent systems, an LLM can assist in interpreting and generating responses in conversation-based AI applications, significantly enhancing the system's capability to mimic human interactions."

    Test cases should validate that the improved response is more detailed, clear, and informative while preserving the accuracy of the original information.
    """,
)

def clean_text(text: str) -> str:
    """
    Cleans the given text by removing or replacing special characters
    while preserving the context.
    """
    replacements = {
        "## ": "",
        "* ": "",
        "**": "",
        ": ": " - ",
        "\n": " ",
        "  ": " ",  # Replaces double spaces with a single space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()

@app.post("/load")
async def load_data(request: LoadDataRequest):
    url = request.url
    if not (url.startswith('http://') or url.startswith('https://')):
        raise HTTPException(status_code=400, detail="Invalid URL. Please include 'http://' or 'https://'.")

    # Send a GET request
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch the Wikipedia page.")

    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main content of the article
    content = soup.find('div', {'id': 'bodyContent'})

    # Extract and clean the text from all paragraphs
    paragraphs = [p.text.strip() for p in content.find_all('p') if p.text.strip()]

    # Embedding paragraphs into vectors
    embeddings = model.encode(paragraphs)

    # Clear existing data in the collection before inserting new documents
    collection.delete_many({})

    # Prepare documents to store in MongoDB
    documents = [{'text': text, 'vector': vector.tolist()} for text, vector in zip(paragraphs, embeddings)]
    collection.insert_many(documents)

    return {"message": "Data loaded successfully, previous data cleared", "total_paragraphs": len(paragraphs)}

@app.post("/query")
async def query_data(request: QueryDataRequest):
    query = request.query
    query_vector = model.encode([query])[0]  # Encode the query to get the vector

    # Find the most similar vectors in MongoDB using cosine similarity
    cursor = collection.find({})
    best_match = None
    max_similarity = -np.inf
    for document in cursor:
        cos_sim = np.dot(query_vector, np.array(document['vector'])) / (
            np.linalg.norm(query_vector) * np.linalg.norm(np.array(document['vector']))
        )
        if cos_sim > max_similarity:
            max_similarity = cos_sim
            best_match = document['text']

    if best_match:
        # Use Gemini to improve the best match response
        chat_session = gemini_model.start_chat(
            history=[
                {"role": "user", "parts": [best_match]},
            ]
        )
        improved_response = chat_session.send_message("INSERT_INPUT_HERE").text
        
        # Clean the generative response
        cleaned_response = clean_text(improved_response)

        return {
            "query": query,
            "best_match": best_match,
            "generative_response": cleaned_response,
            "similarity": max_similarity
        }
    else:
        return {"query": query, "message": "No matching documents found."}
