# store_index.py

from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

# === Step 1: Load environment variables ===
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# === Step 2: Load and preprocess PDF data ===
extracted_data = load_pdf_file(data="Data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# === Step 3: Get HuggingFace embeddings ===
embeddings = download_hugging_face_embeddings()

# === Step 4: Initialize Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # This should match your embedding model output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# === Step 5: Store documents into Pinecone index ===
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"✅ Successfully stored {len(text_chunks)} documents in index '{index_name}'.")
